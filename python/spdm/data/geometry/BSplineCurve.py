
from functools import cached_property
from scipy.interpolate import splev, splprep, splrep, make_interp_spline
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

    def inside(self, *x):
        return NotImplemented

    def points(self, u, *args, **kwargs):
        # if self.is_closed:
        #     return splev(u, self._spl, *args, **kwargs)
        # else:
        #     return splev(u, self._spl, *args, **kwargs)
        return self._spl(u, *args, **kwargs, extrapolate='periodic' if self.is_closed else False).T

    def __call__(self, *args, **kwargs):
        return self.points(*args, **kwargs)

    @cached_property
    def _derivative(self):
        return self._spl.derivative()

    def dl(self, u):
        return np.sqrt(np.inner(self._derivative(u).T))

    def integrate(self, func, p0=None, p1=None):
        return self._spl.integrate(p0 or 0.0, p1 or 1.0, extrapolate='periodic' if self.is_closed else False)
