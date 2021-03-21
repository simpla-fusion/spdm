from functools import cached_property

import numpy as np

from .GeoObject import GeoObject
from .Point import Point

from ...util.logger import logger


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

    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def topology_rank(self):
        return 1

    def inside(self, *x):
        return False

    @cached_property
    def points(self):
        return NotImplemented

    @cached_property
    def dl(self):
        a, b = self.derivative()
        l = self.points[1:]-self.points[:-1]

        a = a[:-1]
        b = b[:-1]

        x = l[:, 0]
        y = l[:, 1]
        m1 = (-a*y+b*x)/(a*x+b*y)

        a = np.roll(a, 1, axis=0)
        b = np.roll(b, 1, axis=0)
        m2 = (-a*y+b*x)/(a*x+b*y)

        r = (2.0*m1**2+2.0*m2**2-m1*m2)/30
        logger.debug((np.mean(m1), np.mean(m2), np.mean(r)))

        res = np.sqrt(x**2+y**2) * (1.0+r)

        return np.hstack([res, res[0]])

    def derivative(self,  *args, **kwargs):
        return NotImplemented

    def pullback(self, func, *args, form=0, **kwargs):
        return NotImplemented


class Line(Curve):
    def __init__(self, *args,   **kwargs) -> None:
        super().__init__(*args, is_closed=False, **kwargs)
