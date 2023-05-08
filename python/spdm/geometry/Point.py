from __future__ import annotations

import collections.abc
import typing
from functools import cached_property
from typing import Callable, Collection, TypeVar

import numpy as np
from spdm.utils.logger import logger

from ..geometry.GeoObject import GeoObject
from .GeoObject import GeoObject


class Point(typing.List[float], GeoObject):
    """ Point
        点，零维几何体
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(args)
        GeoObject.__init__(self, **kwargs)

    @property
    def rank(self) -> int:
        return 0

    @property
    def ndims(self) -> int:
        return len(self)

    def points(self, *args, **kwargs) -> Point:
        return self

    @property
    def measure(self) -> float:
        return 0
