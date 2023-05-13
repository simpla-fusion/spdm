from __future__ import annotations

import os
import typing
from enum import Enum
from functools import lru_cache

import numpy as np

from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix
from ..utils.tags import _not_found_
from ..utils.typing import ArrayType, NumericType
from .Function import Function
from .Node import Node

_T = typing.TypeVar("_T")


class Field(Node, Function, typing.Generic[_T]):
    """ Field
        ---------
        A field is a function that assigns a value to every point of space. The value of the field can be a scalar or a vector.

        一个 _场(Field)_ 是一个 _函数(Function)_，它为空间(默认为多维流形)上的每一点分配一个值。场的值可以是标量或矢量。
    """

    def __init__(self, *args, mesh=None,   **kwargs):

        mesh_desc, coordinates, kwargs = group_dict_by_prefix(kwargs, ("mesh_", "coordinate"))

        super().__init__(*args, **kwargs)

        if mesh is None and len(coordinates) > 0:
            # 获得 coordinates 中 value的共同前缀
            prefix = os.path.commonprefix([v for k, v in coordinates if str(k[0]).isnumeric()])
            if prefix.startswith("../grid/dim"):
                mesh = getattr(self._parent, "grid", None)
                if isinstance(mesh, Node):
                    mesh_type = getattr(self._parent, "grid_type", None)
                    mesh_desc["dim1"] = mesh.dim1
                    mesh_desc["dim2"] = mesh.dim2
                    mesh_desc["volume_element"] = mesh.volume_element
                    mesh_desc["type"] = mesh_type.name if isinstance(mesh_type, Enum) else mesh_type
            elif prefix.startswith("../dim"):
                mesh_type = self._parent.grid_type
                mesh_desc["dim1"] = self._parent.dim1
                mesh_desc["dim2"] = self._parent.dim2
                mesh_desc["type"] = mesh_type.name if isinstance(mesh_type, Enum) else mesh_type

        Function.__init__(self, mesh=mesh)

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
