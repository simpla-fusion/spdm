
import collections
from functools import cached_property
from typing import Union

from spdm.numlib.spline import create_spline

from ..numlib import np
from ..util.logger import logger
from .Curve import Curve


class CubicSplineCurve(Curve):
    def __init__(self,   *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if not isinstance(self._mesh, collections.abc.Sequence):
            raise NotImplementedError(type(self._mesh))
     
        self._spl = create_spline(self._mesh[0], self._points, bc_type="periodic" if self.is_closed else "not-a-knot")

    def points(self,  *args, **kwargs) -> np.ndarray:
        if len(args) == 0:
            return self._points
        else:
            return self._spl(*args, **kwargs)

    @cached_property
    def _derivative(self):
        return self._spl.derivative()

    def derivative(self, *args, **kwargs):
        if len(args) == 0:
            args = self.uv
        res = self._derivative(*args, **kwargs)
        return res[:, 0], res[:, 1]
