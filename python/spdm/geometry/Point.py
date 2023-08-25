from __future__ import annotations


import numpy as np
from spdm.utils.logger import logger

from ..utils.typing import ArrayType, NumericType
from .GeoObject import GeoObject


class Point(GeoObject):
    """ Point
        点，零维几何体
    """

    def __init__(self, *args,   **kwargs) -> None:
        super().__init__(*args, rank=0, ndims=len(args),  ** kwargs)
        self._points = np.array(args)

    @property
    def x(self) -> float: return self._points[0]

    @property
    def y(self) -> float: return self._points[1]

    @property
    def measure(self) -> float:
        return 0

    @property
    def points(self): return self._points
