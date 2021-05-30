
import collections
from functools import cached_property
from typing import Union

from ..numlib import interpolate, np
from ..util.logger import logger
from .Curve import Curve


class CubicSplineCurve(Curve):
    def __init__(self, xy: np.ndarray, u=None, /, *args, is_closed=True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(xy, collections.abc.MutableSequence):
            xy = np.c_[tuple(xy)]
        else:
            xy = np.asarray(xy)

        if u is None:
            u = xy.shape[0]

        if isinstance(u, int):
            u = np.linspace(0, 1.0, u)
        elif not isinstance(u, (collections.abc.Sequence, np.ndarray)):
            u = [u]
        u = np.asarray(u)

        if all(np.isclose(xy[0], xy[-1])):
            is_closed = True
            xy = xy[:-1, :]
            u = u[:-1]
        p_min = np.argmin(u)
        p_max = np.argmax(u)

        if p_min == 0:
            pass
        elif p_min < p_max:
            u[:p_min+1] += 1.0
            u = np.flip(u)
            xy = np.flip(xy, axis=0)
        else:
            # FIXME: need test
            u[p_min+1:] += 1.0

        if is_closed and not all(np.isclose(xy[0], xy[-1])):
            u = np.hstack([u, [u[0]+1.0]])
            xy = np.vstack([xy, xy[:1, :]])
        
        self._spl = interpolate.CubicSpline(u, xy, bc_type="periodic" if is_closed else "not-a-knot")
        self._uv = [u]
        self._xy = xy

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
