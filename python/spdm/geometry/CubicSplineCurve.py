
import collections
from functools import cached_property

import numpy as np
from scipy import constants
from scipy.interpolate import CubicSpline, PPoly
from ..utils.typing import ArrayType
from ..utils.logger import logger
from .Curve import Curve

TWOPI = 2.0*constants.pi


@Curve.register("cubic_spline_curve")
class CubicSplineCurve(Curve):
    def __init__(self, points, uv=None,  **kwargs) -> None:
        super().__init__(**kwargs)
        self._points = points
        self._uv = uv if  uv is None else np.linspace(0, 1, len(self._points))

    def points(self,  *args, **kwargs) -> ArrayType:
        if len(args) == 0:
            return self._points
        else:
            return self._spl(*args, **kwargs)

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
