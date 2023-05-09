from __future__ import annotations

import typing
from functools import lru_cache

from ..utils.typing import NumericType
from .Function import Function


class Field(Function):
    """ Field
        ---------
        A field is a function that assigns a value to every point of space. The value of the field can be a scalar or a vector.

        一个 _场(Field)_ 是一个 _函数(Function)_，它为空间(默认为多维流形)上的每一点分配一个值。场的值可以是标量或矢量。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ppoly: typing.Callable[..., NumericType] = None
        self._derivate = {}

    def __call__(self, *args, **kwargs) -> NumericType:
        return self.__ppoly__()(*args, **kwargs)

    def plot(self, axis, *args,  **kwargs):

        kwargs.setdefault("linewidths", 0.1)

        axis.contour(*self._grid.points,  self.__array__(), **kwargs)

        return axis
