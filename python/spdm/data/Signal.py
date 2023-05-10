import typing
from typing import Any

import numpy as np

from .Field import Field
from .Function import Function
from .Profile import Profile

_T = typing.TypeVar("_T")


class Signal(Profile[_T]):
    """Signal with its time base
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ndims = kwargs.get("ndims", 1)

    @property
    def data(self) -> np.ndarray:
        return super().data

    @property
    def time(self) -> np.ndarray:
        return super().coordinates[-1].__value__()

    def __value__(self) -> Function:
        if self._cache is None:
            self._cache = Function(self.time, self.data)
        return self._cache

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.__value__()(*args, **kwds)


class SignalND(Field[_T]):
    """Signal with its time base
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ndims = kwargs.get("ndims", 1)