from __future__ import annotations

import collections.abc
import typing
from functools import cached_property
from typing import Callable, Collection, TypeVar

import numpy as np
from spdm.utils.logger import logger

from .GeoObject import GeoObject
from ..utils.typing import NumericType, ArrayType


class Point(GeoObject):
    """ Point
        点，零维几何体
    """

    def __init__(self, *args,   **kwargs) -> None:
        super().__init__(*args, rank=0,  ** kwargs)

    @property
    def measure(self) -> float:
        return 0

    def coordinates(self, *uvw) -> NumericType:
        return self._points
