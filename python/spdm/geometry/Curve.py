from functools import cached_property

import numpy as np

from .GeoObject import GeoObject
from .Point import Point
from ..numerical.Function import Function
from ..util.logger import logger


class Curve(GeoObject):
    @staticmethod
    def __new__(cls, *args, type=None, **kwargs):
        if len(args) == 0:
            raise RuntimeError(f"Illegal input! {len(args)}")
        shape = [(len(a) if isinstance(a, np.ndarray) else 1) for a in args]
        if all([s == 1 for s in shape]):
            return object.__new__(Point)
        elif cls is not Curve:
            return object.__new__(cls)
        else:
            # FIXME：　find module
            return object.__new__(Curve)

    def __init__(self,  *args,  **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def topology_rank(self):
        return 1

    @cached_property
    def bbox(self):
        return [[np.min(p) for p in self.xy], [np.max(p) for p in self.xy]]

    def dl(self, u=None, *args, **kwargs):
        if u is None:
            u = self.uv[0]

        L = u[-1]

        u = (u[1:]+u[:-1])*0.5

        a, b = self.derivative(*args, **kwargs)

        x, y = self.xy(*args, **kwargs)

        a = a[:-1]
        b = b[:-1]
        x = x[1:]-x[:-1]
        y = y[1:]-y[:-1]

        m1 = (-a*y+b*x)/(a*x+b*y)

        a = np.roll(a, 1, axis=0)
        b = np.roll(b, 1, axis=0)

        m2 = (-a*y+b*x)/(a*x+b*y)

        d = np.sqrt(x**2+y**2)*(1 + (2.0*m1**2+2.0*m2**2-m1*m2)/30)

        if self.is_closed:
            u = np.hstack([u, [u[0]+L]])
            d = np.hstack([d, [d[0]]])

        return Function(u, d, is_periodic=self.is_closed)

    def integrate(self, fun, u=None):
        dl = self.dl(u)
        f = np.asarray([fun(*p) for p in self.point(dl.x)])
        if self.is_closed:
            return np.sum(((np.roll(f, 1)+f)*dl).view(np.ndarray))*0.5
        else:
            return np.sum((f[1:]+f[:-1])*dl[:-1])*0.5


class Line(Curve):
    def __init__(self, *args,   **kwargs) -> None:
        super().__init__(*args, is_closed=False, **kwargs)
