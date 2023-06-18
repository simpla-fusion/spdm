import functools
import typing
from copy import copy
import numpy as np

from ..utils.logger import logger
from ..utils.typing import (ArrayLike, ArrayType, NumericType, array_type,
                            nTupleType)
from .GeoObject import GeoObject
from .PointSet import PointSet
from .Curve import Curve


class Surface(PointSet):
    """ Surface """

    def __init__(self, *args, uv=None, **kwargs) -> None:
        super().__init__(*args, rank=2, **kwargs)
        self._uv = uv

    def enclose(self, *args) -> bool:
        if not self.is_closed:
            return False
        return super().enclose(*args)

    @property
    def boundary(self) -> Curve:
        return Curve(super().boundary(), is_closed=True)

    @functools.cached_property
    def measure(self) -> float: return np.sum(self.dl)

    def coordinates(self, *uvw, **kwargs) -> ArrayType: return self._spl(*uvw, **kwargs)

    @functools.cached_property
    def _spl(self) -> PPoly:
        return CubicSpline(self._uv, self._points, bc_type="periodic" if self.is_closed else "not-a-knot")

    def derivative(self, *args, **kwargs):
        if len(args) == 0:
            args = [self._uv]
        res = self._derivative(*args, **kwargs)
        return res[:, 0], res[:, 1]

    def remesh(self, u) -> Surface:
        other: Surface = copy(self)
        if isinstance(u, array_type):
            other._uv = u
        elif callable(u):
            other._uv = u(*self.points)
        else:
            raise TypeError(f"illegal type u={type(u)}")
        return other
