
from functools import cached_property
from scipy.interpolate import make_interp_spline
from scipy.integrate import quad
import numpy as np
from ...util.logger import logger
from ..geometry.Curve import Curve


class BSplineCurve(Curve):
    def __init__(self,  *args, is_closed=None, **kwargs) -> None:
        if len(args) != 2:
            raise ValueError(f"Illegal input! len(args)={len(args)}")
        super().__init__(is_closed=is_closed is not None, **kwargs)

        # if self.is_closed:
        #     self._spl, _ = splprep(args, s=0)
        # else:
        #     self._spl = splrep(*args, s=0)
        self._spl = make_interp_spline(np.linspace(0, 1.0, len(args[0])), np.c_[tuple(args)], **kwargs)
        if self.is_closed:
            self._spl.extrapolate = 'periodic'

    def inside(self, *x):
        return NotImplemented

    def points(self, u, *args, **kwargs):
        return self._spl(u, *args, **kwargs).T

    def __call__(self, *args, **kwargs):
        return self.points(*args, **kwargs)

    @cached_property
    def _derivative(self):
        return self._spl.derivative()

    def apply(self, func, *args, **kwargs):
        return func(*self.points(*args, **kwargs))

    def derivative(self, u, *args, **kwargs):
        return self._derivative(u, *args, **kwargs).T

    def dl(self, u, *args, **kwargs):
        return np.linalg.norm(self.derivative(u, *args, **kwargs), axis=0)

    def integrate(self, func, p0=None, p1=None):
        return quad(lambda u: func(*self._spl(u))*self.dl(u), p0 or 0.0, p1 or 1.0, limit=128)

        # return self._spl.integrate(p0 or 0.0, p1 or 1.0, extrapolate='periodic' if self.is_closed else False)
