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

        mesh_desc, kwargs = group_dict_by_prefix(kwargs, "mesh_")

        super().__init__(*args, **kwargs)

        if isinstance(self._mesh, collections.abc.Mapping):
            self._mesh = Mesh(**self._mesh, **mesh_desc)
        elif isinstance(self._mesh, collections.abc.Sequence):
            self._mesh = Mesh(*self._mesh, **mesh_desc)
        elif not isinstance(self._mesh, Mesh):
            logger.warning(f"Field.__init__(): mesh is not a Mesh, but {type(self._mesh)}")

        self._ppoly = {}

    def __mesh__(self) -> Mesh: return self._mesh

    @property
    def mesh(self) -> Mesh: return self._mesh
    """ Field 的 mesh 是 Mesh 类的。
        网格，用来描述流形的几何结构，比如网格的拓扑结构，网格的几何结构，网格的坐标系等。 """

    @property
    def bbox(self): return self.mesh.geometry.bbox

    @property
    def points(self): return self.mesh.points
    """ 定义域 mesh 上的坐标 """

    @property
    def dimensions(self) -> typing.List[ArrayType]: return self.mesh.dims

    @property
    def dims(self) -> typing.List[ArrayType]: return self.mesh.dims

    @property
    def shape(self) -> typing.Tuple[int]: return self.mesh.shape

    @property
    def ndim(self) -> int: return self.mesh.ndim

    def compile(self, *d, force=False,  in_place=True, check_nan=True,   **kwargs) -> Field:
        ppoly = self._fetch_op()
        method = None
        opts = {}

        if isinstance(ppoly, tuple):
            ppoly, opts, *method = ppoly

        if len(d) > 1:
            opts["grid"] = False

        if hasattr(ppoly, "partial_derivative"):
            res = Field[_T](ppoly.partial_derivative(*d), opts=opts,
                            mesh=self.mesh,
                            metadata={"name": f"d{self.name}_d{d}"})
        else:
            ppoly, opts = self.mesh.partial_derivative(self.__array__(), *d)
            res = Field[_T](ppoly, opts=opts,
                            mesh=self.mesh,
                            metadata={"name": f"d{self.name}_d{d}"})

        ppoly,  opts = self._compile()

        if hasattr(ppoly, "antiderivative"):
            res = Field[_T](ppoly.antiderivative(*d), mesh=self.mesh, opts=opts,
                            metadata={"name": f"\int {self.name} d{d}"})
        else:
            ppoly, opts = self.mesh.antiderivative(self.__array__(), *d)
            res = Field[_T](ppoly, mesh=self.mesh, opts=opts,
                            metadata={"name": f"\int {self.name} d{d}"})

    def partial_derivative(self, *d) -> Field[_T]: return self.compile(*d)

    def pd(self, *d) -> Field: return self.partial_derivative(*d)

    def antiderivative(self, *d) -> Field: return self.compile(*d)
