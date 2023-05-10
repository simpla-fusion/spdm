from __future__ import annotations

import typing
from functools import lru_cache

from ..utils.misc import regroup_dict_by_prefix
from ..utils.typing import NumericType
from .Function import Function
from .Node import Node
from .Profile import Profile

_T = typing.TypeVar("_T")


class Field(Profile[_T]):
    """ Field
        ---------
        A field is a function that assigns a value to every point of space. The value of the field can be a scalar or a vector.

        一个 _场(Field)_ 是一个 _函数(Function)_，它为空间(默认为多维流形)上的每一点分配一个值。场的值可以是标量或矢量。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if hasattr(self._parent, "grid"):
            Function.__init__(self, grid=self._parent.grid, **self._appinfo)

    @property
    def metadata(self) -> typing.Mapping[str, typing.Any]:
        return super(Node, self).metadata

    def plot(self, axis, *args,  **kwargs):

        kwargs.setdefault("linewidths", 0.1)

        axis.contour(*self._grid.points,  self.__array__(), **kwargs)

        return axis
