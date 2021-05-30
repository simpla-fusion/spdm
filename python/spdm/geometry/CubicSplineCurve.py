
import collections
from functools import cached_property
from typing import Union

from ..numlib import interpolate, np
from ..util.logger import logger
from .Curve import Curve


class CubicSplineCurve(Curve):
    def __init__(self, u: Union[int, np.ndarray], xy: np.ndarray, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(xy, collections.abc.MutableSequence):
            self._xy = np.c_[tuple(xy)]
        else:

            self._xy = np.asarray(xy)

        if u is None:
            u = self._xy.shape[0]

        if isinstance(u, int):
            u = np.linspace(0, 1.0, u)
        elif not isinstance(u, (collections.abc.Sequence, np.ndarray)):
            u = [u]
        u = np.asarray(u)
        if u.shape[0] == self._xy.shape[0]:
            s = u
        else:
            s = np.linspace(u[0], u[-1], self._xy.shape[0])

        self._spl = interpolate.CubicSpline(s, self._xy, bc_type="periodic" if self.is_closed else "not-a-knot")
        self._uv = [u]

    @cached_property
    def ndims(self):
        return len(self._xy[0])

    @property
    def uv(self):
        return self._uv

    def point(self,  *args, **kwargs):
        if len(args) == 0:
            args = self.uv
        return self._spl(*args, **kwargs)

    @property
    def xy(self) -> np.ndarray:
        return self._xy

    @cached_property
    def _derivative(self):
        return self._spl.derivative()

    def derivative(self, *args, **kwargs):
        if len(args) == 0:
            args = self.uv
        res = self._derivative(*args, **kwargs)
        return res[:, 0], res[:, 1]
