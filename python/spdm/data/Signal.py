import collections.abc
import typing
from enum import Enum
from typing import Any

import numpy as np
from spdm.data.Dict import Dict
from spdm.data.Function import Function
from spdm.data.List import List
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property
from spdm.utils.logger import logger

from .Profile import Profile

_T = typing.TypeVar("_T")


class Signal(Profile, typing.Generic[_T]):
    """Signal with its time base
    """

    def __init__(self, *args, **kwargs):
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
