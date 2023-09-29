from __future__ import annotations

import collections.abc
import functools
import typing
from enum import Enum

from ..mesh.Mesh import Mesh
from ..numlib.calculus import antiderivative, derivative, partial_derivative
from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix
from ..utils.tags import _not_found_
from ..utils.typing import ArrayType, array_type, as_array
from ..utils.tree_utils import merge_tree_recursive
from .Expression import Expression
from .Functor import Functor


class Field(Expression):
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

    def __init__(self, value, *args, mesh=None, metadata=None, parent=None, **kwargs):

        cache = value
        func = None

        if isinstance(cache, (Functor, Expression)):
            func = cache
            cache = None
        elif callable(cache):
            func = Functor(cache)
            cache = None
        else:
            cache = as_array(cache)

        metadata = merge_tree_recursive(metadata, kwargs)

        super().__init__(func, label=metadata.get("label", None) or metadata.get("name", None))

        if mesh is None and len(args) > 0:
            mesh = {"dims": args}
        elif len(args) > 0:
            logger.warning(f"ignore args={args}")

        self._cache = cache
        self._metadata = metadata
        self._parent = parent
        self._mesh = mesh
        self._ppoly = None

    def __repr_svg__(self) -> str:
        from ..view.View import display
        return display(self, output="svg")

    @property
    def mesh(self): return self.__mesh__

    @property
    def __mesh__(self) -> Mesh:
        if isinstance(self._mesh, Mesh):
            return self._mesh

        coordinates, *_ = group_dict_by_prefix(self._metadata, "coordinate", sep=None)

        if self._mesh is None and coordinates is not None:
            coordinates = {int(k): v for k, v in coordinates.items() if k.isdigit()}
            coordinates = dict(sorted(coordinates.items(), key=lambda x: x[0]))

            if all([isinstance(c, str) and c.startswith('../grid') for c in coordinates.values()]):
                o_mesh = getattr(self._parent, "grid", None)
                if isinstance(o_mesh, Mesh):
                    # if self._mesh is not None and len(self._mesh) > 0:
                    #     logger.warning(f"Ignore {self._mesh}")
                    self._mesh = o_mesh
                elif isinstance(o_mesh, collections.abc.Sequence):
                    self._mesh = merge_tree_recursive(self._mesh, {"dims": o_mesh})
                elif isinstance(o_mesh, collections.abc.Mapping):
                    self._mesh = merge_tree_recursive(self._mesh, o_mesh)
                elif o_mesh is not None:
                    raise RuntimeError(f"self._parent.grid is not a Mesh, but {type(o_mesh)}")
            else:
                dims = tuple([(self._parent.get(c) if isinstance(c, str) else c)
                              for c in coordinates.values()])
                self._mesh = merge_tree_recursive(self._mesh, {"dims": dims})

        elif isinstance(self._mesh, Enum):
            self._mesh = {"type": self._mesh.name}

        elif isinstance(self._mesh, str):
            self._mesh = {"type":  self._mesh}

        elif isinstance(self._mesh, collections.abc.Sequence) and all(isinstance(d, array_type) for d in self._mesh):
            self._mesh = {"dims": self._mesh}

        if isinstance(self._mesh, collections.abc.Mapping):
            self._mesh = Mesh(**self._mesh)

        elif not isinstance(self._mesh, Mesh):
            raise RuntimeError(f"self._mesh is not a Mesh, but {type(self._mesh)}")

        return self._mesh

    def __domain__(self, *xargs) -> bool: return self.__mesh__.geometry.enclose(*xargs)

    def __array__(self, *args,  **kwargs) -> ArrayType:
        """ 重载 numpy 的 __array__ 运算符
                若 self._value 为 array_type 或标量类型 则返回函数执行的结果
        """
        value = self._cache

        if (value is None or value is _not_found_):
            value = self.__call__(*self.points)

        if (value is None or value is _not_found_):
            value = None

        return as_array(value)

        # return self._normalize_value(value, *args,  **kwargs)

    @property
    def points(self) -> typing.List[ArrayType]: return self.__mesh__.points

    def __functor__(self) -> Functor:
        if self._func is None:
            self._func = self._interpolate()
        return super().__functor__()

    def _interpolate(self, *args, force=False, **kwargs) -> Functor:
        if self._ppoly is None or force:
            self._ppoly = self.__mesh__.interpolator(self.__array__(), *args, **kwargs)
        return self._ppoly

    def compile(self) -> Field:
        return Field(self._interpolate(), mesh=self.__mesh__, name=f"[{self.__str__()}]")

    def grad(self, n=1) -> Field:
        ppoly = self. __functor__()

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

    def derivative(self, n=1) -> Field:
        return Field(derivative(self. __functor__(), n),  mesh=self.__mesh__, name=f"D_{n}({self})")

    def partial_derivative(self, *d) -> Field:
        return Field(self._interpolate().partial_derivative(*d), mesh=self.__mesh__, name=f"d_{d}({self})")

    def antiderivative(self, *d) -> Field:
        return Field(antiderivative(self. __functor__(), *d),  mesh=self.__mesh__, name=f"I_{d}({self})")

    def d(self, n=1) -> Field: return self.derivative(n)

    def pd(self, *d) -> Field: return self.partial_derivative(*d)

    def dln(self) -> Expression: return self.derivative() / self
