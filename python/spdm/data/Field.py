from __future__ import annotations

import os
import typing

import numpy as np

from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix
from ..utils.tags import _not_found_
from ..utils.typing import ArrayType, NumericType
from .Function import Function
from .Node import Node

_T = typing.TypeVar("_T")


class Field(Node, Function[_T]):
    """ Field
        ---------
        A field is a function that assigns a value to every point of space. The value of the field can be a scalar or a vector.

        一个 _场(Field)_ 是一个 _函数(Function)_，它为空间(默认为多维流形)上的每一点分配一个值。场的值可以是标量或矢量。
    """

    def __init__(self, *args, mesh=None, **kwargs):

        mesh_desc, coordinates, kwargs = group_dict_by_prefix(kwargs, ("mesh_", "coordinate"))

        super().__init__(*args, **kwargs)

        if isinstance(domain, collections.abc.Mapping):
            domain_desc.update(domain)
            domain = None
        elif isinstance(domain, Enum):
            domain_desc.update({"type": domain.name})
            domain = None
        elif isinstance(domain, str):
            domain_desc.update({"type": domain})
            domain = None

        if domain is not None and domain is not _not_found_:
            self._domain = domain
            if len(domain_desc) > 0:
                logger.warning(f"self._mesh is specified, ignore mesh_desc={domain_desc}")
        elif len(domain_desc) == 0:
            self._domain = args if len(args) != 1 else args[0]
        else:
            try:
                from ..mesh.Mesh import Mesh
                self._domain = Mesh(*args, **domain_desc)
            except ModuleNotFoundError:
                raise RuntimeError(f"Can not import Mesh from spdm.mesh.Mesh!")
            except:
                raise RuntimeError(f"Can not create mesh from mesh_desc={domain_desc}")

        if mesh is not None:
            if len(mesh_desc) > 0:
                logger.warning(f"ignore mesh_desc={mesh_desc}")
        elif len(coordinates) > 0:
            # 获得 coordinates 中 value的共同前缀
            dims = {int(k): v for k, v in coordinates.items() if str(k[0]).isdigit()}
            dims = dict(sorted(dims.items(), key=lambda x: x[0]))
            prefix = os.path.commonprefix([*dims.values()])

            if prefix.startswith("../grid/dim"):
                mesh = self._parent.grid
                if isinstance(mesh, Node):
                    mesh_type = getattr(self._parent, "grid_type", None)
                    mesh_desc["dim1"] = getattr(mesh, "dim1", None)
                    mesh_desc["dim2"] = getattr(mesh, "dim2", None)
                    mesh_desc["volume_element"] = mesh.volume_element
                    mesh_desc["type"] = mesh_type
                    mesh = mesh_desc
                elif isinstance(mesh, dict):
                    mesh_desc.update(mesh)
                    mesh = mesh_desc
            elif prefix.startswith("../dim"):
                mesh_type = self._parent.grid_type
                mesh_desc["dim1"] = self._parent.dim1
                mesh_desc["dim2"] = self._parent.dim2
                mesh_desc["type"] = mesh_type
                mesh = mesh_desc
        else:
            mesh = mesh_desc
        Function.__init__(self, domain=mesh)

    @ property
    def metadata(self) -> typing.Mapping[str, typing.Any]: return super(Node, self).metadata

    @ property
    def data(self) -> ArrayType: return self.__array__()

    @ property
    def domain(self) -> typing.Tuple[typing.Tuple[NumericType, NumericType], typing.Tuple[NumericType, NumericType]]:

        if isinstance(self._mesh, np.ndarray):
            return tuple([np.min(self._mesh), np.max(self._mesh)])
        elif isinstance(self._mesh, tuple):
            return tuple([[np.min(d) for d in self._mesh], [np.max(d) for d in self._mesh]])
        elif hasattr(self._mesh, "bbox"):
            return self._mesh.bbox
        else:
            raise RuntimeError(f"Cannot get bbox of {self._mesh}")

    def __array__(self) -> ArrayType:
        value = Node.__value__(self)
        if value is None or value is _not_found_:
            value = Function.__array__(self)
            self._cache = value
        return value

    def __ppoly__(self, *dx: int) -> typing.Callable[..., NumericType]:
        """ 返回 PPoly 对象
            TODO:
            - support JIT compile
            - 优化缓存
            - 支持多维插值
            - 支持多维求导，自动微分 auto diff
            -
        """
        fun = self._ppoly_cache.get(dx, None)
        if fun is not None:
            return fun

        if len(dx) == 0:
            fun = None

            if len(self._domain) == 1 and isinstance(self._domain[0], np.ndarray):
                fun = InterpolatedUnivariateSpline(self._domain[0], self._array)
            elif len(self._domain) == 2 and all(isinstance(d, np.ndarray) for d in self._domain):
                fun = RectBivariateSpline(*self._domain, self._array)
            else:
                raise RuntimeError(
                    f"Can not convert Function to PPoly! value={type(self._array)} domain={self._domain} ")

        else:
            ppoly = self.__ppoly__()

            if all(d < 0 for d in dx):
                if hasattr(ppoly.__class__, 'antiderivative'):
                    fun = self.__ppoly__().antiderivative(*dx)
                elif hasattr(self._domain, "antiderivative"):
                    fun = self._domain.antiderivative(self.__array__(), *dx)
            elif all(d >= 0 for d in dx):
                if hasattr(ppoly.__class__, 'partial_derivative'):
                    fun = self.__ppoly__().partial_derivative(*dx)
                elif hasattr(self._domain, "partial_derivative"):
                    fun = self._domain.partial_derivative(self.__array__(), *dx)
            else:
                raise RuntimeError(f"{dx}")

        self._ppoly_cache[dx] = fun

        return fun

    def plot(self, axis, *args,  **kwargs):

        kwargs.setdefault("linewidths", 0.1)

        axis.contour(*self._mesh.points,  self.__array__(), **kwargs)

        return axis
