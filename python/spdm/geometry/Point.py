from __future__ import annotations


import numpy as np
from spdm.utils.logger import logger

from ..utils.typing import ArrayType, NumericType
from .GeoObject import GeoObject


class Point(GeoObject):
    """Point
    点，零维几何体
    """

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 1:
            if isinstance(args[0], (list, tuple)):
                args = args[0]
            elif isinstance(args[0], dict):
                args = [*args[0].values()]                
        super().__init__(*args, rank=0, ndims=len(args), **kwargs)
        self._coord = np.array(args)

    @property
    def x(self) -> float:
        return self._coord[0]

    @property
    def y(self) -> float:
        return self._coord[1]

    @property
    def measure(self) -> float:
        return 0

    @property
    def points(self):
        return self._coord
