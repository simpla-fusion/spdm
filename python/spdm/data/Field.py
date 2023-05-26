from __future__ import annotations

import collections.abc
import typing
from enum import Enum

import numpy as np

from ..mesh.Mesh import Mesh
from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix
from ..utils.tags import _not_found_
from ..utils.typing import ArrayType, NumericType, array_type, numeric_type
from .Expression import Expression
from .Node import Node

_T = typing.TypeVar("_T")


class Field(Expression, Node, typing.Generic[_T]):
    """ Field
        ---------
        Field 是 Function 在流形（manifold/Mesh）上的推广， 用于描述流形上的标量场，矢量场，张量场等。

        Field 所在的流形记为 mesh ，可以是任意维度的，可以是任意形状的，可以是任意拓扑的，可以是任意坐标系的。

        Mesh 网格描述流形的几何结构，比如网格的拓扑结构，网格的几何结构，网格的坐标系等。

        Field 与 Function的区别：
            - Function 的 mesh 是 多个一维数组表示dimensions/axis
            - Field 的 mesh 是 Mesh，可以表示复杂流形上的场等。

        当 Field 不做为 DTree 的节点时， 应直接由Function继承  Field(Function[_T])

    """

    def __init__(self, value: NumericType | Expression,  *args, name=None, **kwargs):

        op = None

        if hasattr(value, "__entry__"):
            entry = value.__entry__()
            value = None
        elif isinstance(value, collections.abc.Mapping):
            entry = value
            value = None
        elif isinstance(value, Expression) or callable(value) or (isinstance(value, tuple) and callable(value[0])):
            op = value
            value = None
            entry = None
        else:
            op = None
            entry = None

        mesh,  kwargs = group_dict_by_prefix(kwargs,  "mesh")

        Node.__init__(self, entry, **kwargs)

        Expression.__init__(self, op, name=name if name is not None else self._metadata.get("name", None))

        self._value = value

        if isinstance(mesh, Enum):
            self._mesh = {"type": mesh.name}
        elif isinstance(mesh, str):
            self._mesh = {"type":  mesh}
        elif isinstance(mesh, collections.abc.Sequence) and all(isinstance(d, array_type) for d in mesh):
            self._mesh = {"dims": mesh}
        else:
            self._mesh = mesh if mesh is not None else {}

        if isinstance(self._mesh, collections.abc.Mapping) and len(args) > 0:
            self._mesh["dims"] = args
        elif len(args) > 0:
            raise RuntimeError(f"ignore args={args}")

    @property
    def mesh(self): return self.__mesh__

    @property
    def __mesh__(self) -> Mesh:
        if isinstance(self._mesh, Mesh):
            return self._mesh

        mesh_desc, metadata = group_dict_by_prefix(self._metadata, "mesh")

        if self._mesh is None:
            self._mesh = mesh_desc
        elif isinstance(self._mesh, collections.abc.Mapping) and mesh_desc is not None:
            self._mesh.update(mesh_desc)
        # else:
        #     raise TypeError(f"self._mesh={self._mesh} is not a Mapping")

        if isinstance(self._parent, Node):
            coordinates, *_ = group_dict_by_prefix(metadata, "coordinate", sep=None)
            coordinates = {int(k): v for k, v in coordinates.items() if k.isdigit()}
            coordinates = dict(sorted(coordinates.items(), key=lambda x: x[0]))

            if len(coordinates) == 0:
                pass
            elif all([isinstance(c, str) and c.startswith('../grid') for c in coordinates.values()]):
                o_mesh = getattr(self._parent, "grid", None)
                if isinstance(o_mesh, Mesh):
                    if len(self._mesh) > 0:
                        logger.warning(f"Ignore {self._mesh}")
                    self._mesh = o_mesh
                elif isinstance(o_mesh, collections.abc.Sequence):
                    self._mesh["dims"] = o_mesh
                elif isinstance(o_mesh, collections.abc.Mapping):
                    self._mesh.update(o_mesh)
                elif o_mesh is not None:
                    raise RuntimeError(f"self._parent.grid is not a Mesh, but {type(o_mesh)}")
            else:
                if self._mesh is None:
                    self._mesh = {}
                self._mesh["dims"] = tuple([(self._parent._find_node_by_path(c, prefix="../") if isinstance(c, str) else c)
                                            for c in coordinates.values()])

        if isinstance(self._mesh, collections.abc.Mapping):
            self._mesh = Mesh(**self._mesh)

        if not isinstance(self._mesh, Mesh):
            raise RuntimeError(f"self._mesh is not a Mesh, but {type(self._mesh)}")

        return self._mesh

    def __domain__(self, *xargs) -> bool: return self.__mesh__.geometry.enclose(*xargs)

    def __value__(self) -> ArrayType:
        if self._value is not None:
            return self._value
        value = Node.__value__(self)

        if isinstance(value, Expression) or callable(value):
            op = value
            value = op(*self.points)
            if self._op is None:
                self._op = op
            elif len(self._children) == 0:
                self._children = (op,)
            else:
                raise ValueError("op is not None, but children is not empty")

        return value

    def __array__(self, dtype=None, *args,  **kwargs) -> ArrayType:
        """ 重载 numpy 的 __array__ 运算符
                若 self._value 为 array_type 或标量类型 则返回函数执行的结果
        """
        res = self.__value__()

        if res is None or res is _not_found_ and self.callable:
            res = self._value = self._eval(*self.points)

        if isinstance(res, numeric_type):
            res = np.asarray(res, dtype=self.__type_hint__ if dtype is None else dtype)
        else:
            raise TypeError(f" Can not get value {(res)}! fun={self.__str__()}")
        return res

    def _compile(self, *d,   **kwargs) -> Field:
        # if hasattr(self.__mesh__, "dims"):  # as rectlinear grid
        #     return super()._compile(*d,  **kwargs)

        # FIMXE: STILL WORKING! NOT FINISH!!!

        value = self.__value__()

        if value is None or value is _not_found_ or len(value) == 0 and self.callable:
            value = np.asarray(super()._eval(*self.__mesh__.points))

        if not isinstance(value, array_type):
            raise RuntimeError(f"Function.compile() incorrect value {self.__str__()} value={value}   ")
        elif len(value.shape) == 0 or value.shape == (1,):
            # 如果value是标量，无法插值，则返回 value 作为常函数
            return value

        if tuple(value.shape) != tuple(self.__mesh__.shape):
            raise NotImplementedError(f"{value.shape}!={self.__mesh__.shape}")

        if self.__mesh__.ndim == 1 and len(d) <= 0:
            if hasattr(self.__mesh__, "derivative"):
                return self.__mesh__.derivative(d, value)
            else:
                return super()._compile(*d, **kwargs)
        elif self.__mesh__.ndim == 2 and d == (1,):
            return self._compile(1, 0, **kwargs), self._compile(0, 1, **kwargs)
        elif self.__mesh__.ndim == 3 and d == (1,):
            return self._compile(1, 0, 0, **kwargs), self._compile(0, 1, 0, **kwargs), self._compile(0, 0, 1, **kwargs)
        elif len(d) == 1:
            raise NotImplementedError(f"ndim={self.__mesh__.ndim} d={d}")
        elif len(d) != self.__mesh__.ndim:
            raise RuntimeError(f"Illegal! ndim={self.__mesh__.ndim} d={d}")
        elif all(v == 0 for v in d) and hasattr(self.__mesh__, "interpolator"):
            return self.__mesh__.interpolator(value)
        elif all(v >= 0 for v in d) and hasattr(self.__mesh__, "partial_derivative"):
            return self.__mesh__.partial_derivative(d, value)
        elif all(v <= 0 for v in d) and hasattr(self.__mesh__, "antiderivative"):
            return self.__mesh__.antiderivative([-v for v in d], value)
        else:
            raise NotImplementedError(f"TODO: {d}")

    def grad(self, n=1) -> Field:
        ppoly = self._compile()

        if isinstance(ppoly, tuple):
            ppoly, opts = ppoly
        else:
            opts = {}

        if self.__mesh__.ndim == 2 and n == 1:
            return Field[typing.Tuple[_T, _T]]((ppoly.partial_derivative(1, 0),
                                                ppoly.partial_derivative(0, 1)),
                                               mesh=self.__mesh__,
                                               name=f"\\nabla({self.__str__()})", **opts)
        elif self.__mesh__.ndim == 3 and n == 1:
            return Field[typing.Tuple[_T, _T, _T]]((ppoly.partial_derivative(1, 0, 0),
                                                    ppoly.partial_derivative(0, 1, 0),
                                                    ppoly.partial_derivative(0, 0, 1)),
                                                   mesh=self.__mesh__,
                                                   name=f"\\nabla({self.__str__()})", **opts)
        elif self.__mesh__.ndim == 2 and n == 2:
            return Field[typing.Tuple[_T, _T, _T]]((ppoly.partial_derivative(2, 0),
                                                    ppoly.partial_derivative(0, 2),
                                                    ppoly.partial_derivative(1, 1)),
                                                   mesh=self.__mesh__,
                                                   name=f"\\nabla^{n}({self.__str__()})", **opts)
        else:
            raise NotImplemented(f"TODO: ndim={self.__mesh__.ndim} n={n}")

    def partial_derivative(self, *d) -> Field[_T]:
        if len(d) == 0:
            d = (1,)
        return Field[_T](self._compile(*d), mesh=self.__mesh__, name=f"d_{d}({self.__str__()})")

    def pd(self, *d) -> Field[_T]: return self.partial_derivative(*d)
