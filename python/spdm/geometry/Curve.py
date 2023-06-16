import abc
import typing
from functools import cached_property

import numpy as np
from ..utils.logger import logger
from ..utils.typing import ArrayLike, ArrayType, NumericType, nTupleType
from .GeoObject import GeoObject


class Curve(GeoObject):
    """ Curve
        曲线，一维几何体
    """

    def __init__(self, *args, is_closed=False, **kwargs) -> None:
        super().__init__(*args, rank=1, **kwargs)
        self._is_closed = is_closed

    # @abc.abstractproperty
    # def is_convex(self) -> bool: return True

    @property
    def is_closed(self) -> bool: return self._is_closed

    @cached_property
    def dl(self) -> ArrayType:
        x, y = self.points
        a, b = self.derivative()

        # a = a[:-1]
        # b = b[:-1]
        dx = x[1:]-x[:-1]
        dy = y[1:]-y[:-1]

        m1 = (-a[:-1]*dy+b[:-1]*dx)/(a[:-1]*dx+b[:-1]*dy)

        # a = np.roll(a, 1, axis=0)
        # b = np.roll(b, 1, axis=0)

        m2 = (-a[1:]*dy+b[1:]*dx)/(a[1:]*dx+b[1:]*dy)

        return np.sqrt(dx**2+dy**2)*(1 + (2.0*m1**2+2.0*m2**2-m1*m2)/30)

    @cached_property
    def measure(self) -> float: return np.sum(self.dl)

    def integral(self, func: typing.Callable) -> float:
        x, y = self.points
        val = func(x, y)

        # c_pts = self.points((self._mesh[0][1:] + self._mesh[0][:-1])*0.5)

        return np.sum(0.5*(val[:-1]+val[1:]) * self.dl)
