from __future__ import annotations

import os
import typing
from enum import Enum
from functools import lru_cache

import numpy as np

from ..mesh.Mesh import Mesh
from ..utils.logger import logger
from ..utils.misc import regroup_dict_by_prefix
from ..utils.typing import ArrayType, NumericType
from .Function import Function
from .Node import Node
from .Profile import Profile
from ..utils.tags import _not_found_
_T = typing.TypeVar("_T")


class Field(Node, Function, typing.Generic[_T]):
    """ Field
        ---------
        A field is a function that assigns a value to every point of space. The value of the field can be a scalar or a vector.

        一个 _场(Field)_ 是一个 _函数(Function)_，它为空间(默认为多维流形)上的每一点分配一个值。场的值可以是标量或矢量。
    """

    def __init__(self, *args, mesh=None, **kwargs):
        Mesh_desc, kwargs = regroup_dict_by_prefix(kwargs, "Mesh")
        coordinates, kwargs = regroup_dict_by_prefix(kwargs, "coordinate", keep_prefix=True, sep='')
        super().__init__(*args, **kwargs)

        if isinstance(mesh, Mesh):
            pass
        elif isinstance(Mesh_desc, dict) and len(Mesh_desc) > 0:
            mesh = Mesh(**Mesh_desc)
        elif len(coordinates) > 0:
            # 获得 coordinates 中 value的共同前缀
            prefix = os.path.commonprefix([*coordinates.values()])
            if prefix.startswith("../Mesh/dim"):
                mesh = getattr(self._parent, "Mesh", None)
                if isinstance(mesh, Node):
                    mesh_type = getattr(self._parent, "mesh_type", None)
                    mesh = Mesh(mesh.dim1, mesh.dim2, mesh.volume_element, type=mesh_type)
            elif prefix.startswith("../dim"):
                mesh = Mesh(self._parent.dim1, self._parent.dim2, self._parent._parent.mesh_type)
            else:
                mesh = Mesh()

        if not isinstance(mesh, Mesh):
            mesh_type = getattr(self._parent, "mesh_type", None)

            if not isinstance(mesh_type, str):
                mesh_type = getattr(mesh_type, "name", None)

            mesh = Mesh(mesh, type=mesh_type)

        Function.__init__(self, Mesh=mesh)

    @property
    def metadata(self) -> typing.Mapping[str, typing.Any]:
        return super(Node, self).metadata

    @property
    def data(self) -> ArrayType: return self.__array__()

    def __array__(self) -> ArrayType:
        if self._cache is None:
            self._cache = self._entry.__value__()
        if self._cache is None or self._cache is _not_found_:
            self._cache = Function.__array__(self)
        return self._cache

    def plot(self, axis, *args,  **kwargs):

        kwargs.setdefault("linewidths", 0.1)

        axis.contour(*self._Mesh.points,  self.__array__(), **kwargs)

        return axis
