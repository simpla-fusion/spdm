from __future__ import annotations

import collections.abc
import os
import typing
from enum import Enum
from functools import cached_property

import numpy as np

from ..geometry.GeoObject import GeoObject
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

    def __init__(self, value: NumericType | Expression, *args, mesh=None, **kwargs):
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

        mesh_desc, coordinates, opts, kwargs = group_dict_by_prefix(kwargs, ("mesh_", "coordinate", "op_"))

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

        if isinstance(mesh, Mesh):
            self._mesh = mesh
            if len(mesh_desc) > 0 or len(args) > 0:
                logger.warning(f"Ignore mesh_desc={mesh_desc} and args={args}!")
        else:
            if isinstance(mesh, collections.abc.Mapping):
                mesh_desc.update(mesh)
                mesh = None
            elif isinstance(mesh, Enum):
                mesh_desc.update({"type": mesh.name})
                mesh = None
            elif isinstance(mesh, str):
                mesh_desc.update({"type": mesh})
                mesh = None
            elif mesh is not None:
                raise TypeError(f"Illegal mesh typ {type(mesh)} !")

            if len(coordinates) > 0:
                if len(args) > 0:
                    raise RuntimeError(f"Coordiantes is defined twice!  len(args)={len(args)}")

                # 获得 coordinates 中的共同前缀
                coord_path = {int(k): v for k, v in coordinates.items() if str(k[0]).isdigit()}

                coord_path = dict(sorted(coord_path.items(), key=lambda x: x[0]))

                if len(coord_path) > 0:
                    args = tuple([(slice(None) if (c == "1...N")
                                   or not isinstance(c, str) else self._find_node_by_path(c)) for c in coord_path.values()])

                    if coord_path[1].startswith("../grid/dim"):
                        mesh_desc["type"] = self._parent.grid_type
                    elif coord_path[1].startswith("../dim"):
                        mesh_desc["type"] = self._parent.grid_type

            self._mesh = Mesh(*args, **mesh_desc)

    @property
    def mesh(self) -> Mesh: return self._mesh
    """ 网格，用来描述流形的几何结构，比如网格的拓扑结构，网格的几何结构，网格的坐标系等。 """

    @property
    def domain(self) -> GeoObject: return self.mesh.geometry
    """ 函数的定义域，返回几何体 """

    @property
    def bbox(self): return self.mesh.geometry.bbox

    def __value__(self) -> ArrayType:
        value = Node.__value__(self)
        if isinstance(value, np.ndarray):
            return value
        
        if value is _not_found_:
            if self._op is not None:
                value = super().__call__(*self.mesh.points)
            else:
                raise RuntimeError(f"Field.__array__(): value is not found!")

        if not isinstance(value, np.ndarray):
            value = np.asarray(value, dtype=self.__type_hint__)
        self._cache = value
        return value

    def __array__(self) -> typing.Any: return self.__value__()

    def __call__(self, *args, ** kwargs) -> NumericType:
        if self._op is None:
            self._op = self.mesh.interpolator(self.__array__())

        if len(args) == 0:
            args = self.mesh.points
        return super().__call__(*args, ** kwargs)

    def partial_derivative(self, *d) -> Field:
        if hasattr(self._op, "partial_derivative"):
            return Field(self._op.partial_derivative(*d), self.mesh)
        else:
            return Field(self.mesh.partial_derivative(self.__array__(), *d), self.mesh)

    def pd(self, *d) -> Field: return self.partial_derivative(*d)

    def antiderivative(self, *d) -> Field:
        if hasattr(self._op, "antiderivative"):
            return Field(self._op.antiderivative(*d), self.mesh)
        else:
            return Field(self.mesh.antiderivative(self.__array__(), *d), self.mesh)
