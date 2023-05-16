
import collections
import typing
from functools import cached_property

import numpy as np
from scipy import constants
from scipy.interpolate import CubicSpline, PPoly

from ..utils.logger import logger
from ..utils.typing import ArrayType
from .Curve import Curve

TWOPI = 2.0*constants.pi


@Curve.register("cubic_spline_curve")
class CubicSplineCurve(Curve):
    def __init__(self, points, uv=None,  **kwargs) -> None:
        super().__init__(points, **kwargs)
        self._uv = uv if uv is None else np.linspace(0, 1, len(self._points))

    def coordinates(self, *uvw, **kwargs) -> ArrayType:
        if len(uvw) == 0:
            return self._points
        else:
            return self._spl(*uvw, **kwargs)

    @cached_property
    def _spl(self) -> PPoly:
        return CubicSpline(self._uv, self._points, bc_type="periodic" if self.is_closed else "not-a-knot")

    @cached_property
    def _derivative(self):
        return self._spl.derivative()

    def derivative(self, *args, **kwargs):
        if len(args) == 0:
            args = [self._uv]
        res = self._derivative(*args, **kwargs)
        return res[:, 0], res[:, 1]
