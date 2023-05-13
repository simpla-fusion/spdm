from __future__ import annotations

import collections.abc
import typing
from functools import cached_property
from typing import Callable, Collection, TypeVar

import numpy as np
from spdm.utils.logger import logger

from .GeoObject import GeoObject
from ..utils.typing import NumericType


class Point(GeoObject):
    """ Point
        点，零维几何体
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(rank=0, ndim=len(args), ** kwargs)
        self._data: typing.Tuple[float, ...] = args

    def __getitem__(self, index: int) -> float:
        return self._data[index]

    def points(self, *uv, **kwargs) -> typing.Tuple[NumericType, ...]:
        if len(uv) == 0 or isinstance(uv[0], float):
            return self._data
        elif isinstance(uv[0], np.ndarray):
            shape = [len(d) for d in uv]
            return np.tile(self._data, shape + [1])
        else:
            raise RuntimeError(f"illegal {uv}")

    @property
    def measure(self) -> float:
        return 0
