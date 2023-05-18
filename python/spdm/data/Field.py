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
from .Function import Function
from .Expression import Expression
from .Profile import Profile
from .Node import Node

_T = typing.TypeVar("_T")


class Field(Profile[_T]):
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

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(self._mesh, collections.abc.Mapping):
            self._mesh = Mesh(**self._mesh)
        elif isinstance(self._mesh, collections.abc.Sequence):
            self._mesh = Mesh(*self._mesh)
        elif not isinstance(self._mesh, Mesh):
            logger.warning(f"Field.__init__(): mesh is not a Mesh, but {type(self._mesh)}")

    @property
    def mesh(self) -> Mesh: return self._mesh
    """ Field 的 mesh 是 Mesh 类的。
        网格，用来描述流形的几何结构，比如网格的拓扑结构，网格的几何结构，网格的坐标系等。 """

    @property
    def bbox(self): return self.mesh.geometry.bbox

    @property
    def points(self): return self.mesh.points
    """ 定义域 mesh 上的坐标 """

    def __array__(self) -> ArrayType: return self.__value__()

    def _eval(self, *args, **kwargs) -> NumericType:
        if self._op is None:
            self._op = self.mesh.interpolator(self.__array__())

        if len(args) == 0:
            args = self.points

        if all([isinstance(a, np.ndarray) for a in args]):
            shape = args[0].shape
            return super()._eval(*[a.ravel() for a in args],  **kwargs).reshape(shape)
        else:
            return super()._eval(*args,  **kwargs)

    def compile(self) -> Field[_T]:
        return Field[_T](self.__array__(), mesh=self.mesh, metadata=self._metadata)

    def partial_derivative(self, *d) -> Field[_T]:
        if hasattr(self._op, "partial_derivative"):
            res = Field[_T](self._op.partial_derivative(*d),  mesh=self.mesh,
                            metadata={"name": f"d{self.name}_d{d}"})
        else:
            res = Field[_T](self.mesh.partial_derivative(self.__array__(), *d), mesh=self.mesh,
                            metadata={"name": f"d{self.name}_d{d}"})
        return res

    def pd(self, *d) -> Field: return self.partial_derivative(*d)

    def antiderivative(self, *d) -> Field:
        if hasattr(self._op, "antiderivative"):
            res = Field[_T](self._op.antiderivative(*d), mesh=self.mesh,
                            metadata={"name": f"\int {self.name} d{d}"})
        else:
            res = Field[_T](self.mesh.antiderivative(self.__array__(), *d), mesh=self.mesh,
                            metadata={"name": f"\int {self.name} d{d}"})
        return res
