
import collections
from functools import cached_property

from ..util.logger import logger
from ..util.numlib import interpolate, np
from .Curve import Curve


class CubicSplineCurve(Curve):
    def __init__(self, u, xy, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(xy, collections.abc.MutableSequence):
            xy = np.c_[tuple(xy)]
        self._spl = interpolate.CubicSpline(u, xy, bc_type="periodic" if self.is_closed else "not-a-knot")

    @cached_property
    def uv(self):
        return [self._spl.x]

    def point(self,  *args, **kwargs):
        if len(args) == 0:
            args = self.uv
        return self._spl(*args, **kwargs)

    @cached_property
    def _derivative(self):
        return self._spl.derivative()

    def derivative(self, *args, **kwargs):
        if len(args) == 0:
            args = self.uv
        res = self._derivative(*args, **kwargs)
        return res[:, 0], res[:, 1]
