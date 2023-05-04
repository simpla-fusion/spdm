from __future__ import annotations

import collections.abc
import typing
from functools import cached_property
from typing import Callable, Collection, TypeVar

import numpy as np
from spdm.utils.logger import logger

from ..geometry.GeoObject import GeoObject
from .GeoObject import GeoObject


class Point(np.ndarray, GeoObject):
    """ Point
        点，零维几何体
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args)
        GeoObject.__init__(self, **kwargs)

    @property
    def rank(self) -> int:
        return 0

    def points(self, *args, **kwargs) -> Point:
        return self

    @property
    def measure(self) -> float:
        return 0
