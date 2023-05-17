from __future__ import annotations

import collections.abc
import typing
from enum import Enum

import numpy as np

from ..mesh.Mesh import Mesh
from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix
from ..utils.tags import _not_found_
from ..utils.typing import ArrayType, NumericType
from .Expression import Expression
from .Node import Node

_T = typing.TypeVar("_T")


class Field(Node, Expression[_T]):
    """ Field
        ---------
        A field is a function that assigns a value to every point of space.
        Field 是 Function 在流形（manifold）上的推广， 用于描述流形上的标量场，矢量场，张量场等。

        Mesh：网格，用来描述流形的几何结构，比如网格的拓扑结构，网格的几何结构，网格的坐标系等。

    """

    def __init__(self, value: NumericType | Expression, *args, domain=None, **kwargs):
        """
            Parameters
            ----------
            value : NumericType
                函数的值
            domain : typing.List[ArrayType]
                函数的定义域
            args : typing.Any
                位置参数, 用于与mesh_*，coordinate* 一起构建 mesh
            kwargs : typing.Any
                命名参数，
                    mesh_*      : 用于传递给网格的参数
                    coordinate* : 给出各个坐标轴的path
                    op_*        : 用于传递给运算符的参数
                    *           : 用于传递给 Node 的参数

        """

        domain_desc, coordinates, opts, kwargs = group_dict_by_prefix(kwargs, ("mesh_", "coordinate", "op_"))

        if isinstance(value, Expression):
            Expression.__init__(self, value, **opts)
            value = None
        elif callable(value):
            if "op" in kwargs:
                raise RuntimeError(f"Can not specify both value and op!")
            Expression.__init__(self, op=value, **opts)
            value = None
        else:
            Expression.__init__(self, **opts)

        Node.__init__(self, value, **kwargs)

        if isinstance(domain, Mesh):
            self._mesh = domain
            if len(domain_desc) > 0 or len(args) > 0:
                logger.warning(f"Ignore domain={domain_desc} and args={args}!")
        else:
            if isinstance(domain, collections.abc.Mapping):
                domain_desc.update(domain)
                domain = None
            elif isinstance(domain, Enum):
                domain_desc.update({"type": domain.name})
                domain = None
            elif isinstance(domain, str):
                domain_desc.update({"type": domain})
                domain = None
            elif domain is not None:
                raise TypeError(f"Illegal mesh typ {type(domain)} !")

            if len(coordinates) > 0:
                if len(args) > 0:
                    raise RuntimeError(f"Coordiantes is defined twice!  len(args)={len(args)}")

                # 获得 coordinates 中的共同前缀
                coord_path = {int(k): v for k, v in coordinates.items() if str(k[0]).isdigit()}

                coord_path = dict(sorted(coord_path.items(), key=lambda x: x[0]))

                if len(coord_path) > 0:
                    if all(p.startswith("../grid/dim") for p in coord_path.values()):
                        self._mesh = self._parent.grid
                    else:
                        args = tuple([(slice(None) if (c == "1...N")
                                       or not isinstance(c, str) else self._find_node_by_path(c)) for c in coord_path.values()])
                        self._mesh = Mesh(*args, **domain_desc)

    @property
    def domain(self) -> Mesh: return self.mesh
    """ 定义域， Field 的定义域为 Mesh """

    @property
    def mesh(self) -> Mesh: return self._mesh
    """ alias of domain
        网格，用来描述流形的几何结构，比如网格的拓扑结构，网格的几何结构，网格的坐标系等。 """

    @property
    def bbox(self): return self.mesh.geometry.bbox

    def __value__(self) -> ArrayType:
        value = Node.__value__(self)
        if isinstance(value, np.ndarray):
            return value

        if not isinstance(value, np.ndarray) and not value:
            if self._op is not None:
                value = super().__call__(*self.mesh.xyz)
            else:
                raise RuntimeError(f"Field.__array__(): value is not found!")

        if not isinstance(value, np.ndarray):
            value = np.asarray(value, dtype=self.__type_hint__)
        self._cache = value
        return value

    def __array__(self) -> typing.Any: return self.__value__()

    def __call__(self, *args, **kwargs) -> NumericType:
        if self._op is None:
            self._op = self.mesh.interpolator(self.__array__())
            self._opts.setdefault("grid", False)

        if len(args) == 0:
            args = self.mesh.xyz

        if all([isinstance(a, np.ndarray) for a in args]):
            shape = args[0].shape
            return super().__call__(*[a.ravel() for a in args],  **kwargs).reshape(shape)
        else:
            return super().__call__(*args,  **kwargs)

    def compile(self, *args, **kwargs) -> Field[_T]:
        if len(args) > 0:
            raise NotImplementedError(f"Field.compile() does not support args={args}!")
        v = self.__value__()
        if v is None or v is _not_found_:
            raise RuntimeError(f"Field.compile() failed!")
        res = Field[_T](v, domain=self.mesh, op_grid=False, **kwargs)
        res._opts.update(self._opts)
        return res

    def partial_derivative(self, *d) -> Field[_T]:
        if hasattr(self._op, "partial_derivative"):
            res = Field(self._op.partial_derivative(*d),  domain=self.mesh)
        else:
            res = Field(self.mesh.partial_derivative(self.__array__(), *d),  domain=self.mesh)
        res._opts.update(self._opts)
        return res

    def pd(self, *d) -> Field: return self.partial_derivative(*d)

    def antiderivative(self, *d) -> Field:
        if hasattr(self._op, "antiderivative"):
            res = Field[_T](self._op.antiderivative(*d), domain=self.mesh)
        else:
            res = Field[_T](self.mesh.antiderivative(self.__array__(), *d),  domain=self.mesh)
        res._opts.update(self._opts)
        return res
