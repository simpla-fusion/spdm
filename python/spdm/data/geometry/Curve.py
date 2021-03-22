from functools import cached_property

import numpy as np

from .GeoObject import GeoObject
from .Point import Point
from ..Function import Function
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

    def __init__(self,  *args,  **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def topology_rank(self):
        return 1

    @cached_property
    def bbox(self):
        return [[np.min(p) for p in self.xy], [np.max(p) for p in self.xy]]

    def dl(self,   *args, **kwargs):
        if len(args) == 0:
            args = self.uv
        a, b = self.derivative(*args, **kwargs)
        x, y = self.xy(*args, **kwargs)
        m1 = (-a*y+b*x)/(a*x+b*y)

        a = np.roll(a, 1, axis=0)
        b = np.roll(b, 1, axis=0)
        m2 = (-a*y+b*x)/(a*x+b*y)

        r = (2.0*m1**2+2.0*m2**2-m1*m2)/30
        if np.mean(r) > 1000:
            logger.debug(a.shape)

        return Function(*args, np.sqrt(x**2+y**2)*(1 + r), is_period=self.is_closed)


class Line(Curve):
    def __init__(self, *args,   **kwargs) -> None:
        super().__init__(*args, is_closed=False, **kwargs)
