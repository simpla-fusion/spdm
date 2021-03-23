from functools import cached_property

import numpy as np
import scipy.interpolate

from ..util.logger import logger
from .Quantity import Quantity


class Function(Quantity):
    @staticmethod
    def __new__(cls,  *args, is_periodic=False, **kwargs):
        if cls is not Function:
            return Quantity.__new__(cls, *args, **kwargs)

        obj = None

        if len(args) != 2:
            pass
        elif isinstance(args[1], Function):
            d = args[1](args[0])
            ppoly = args[1]._ppoly
        elif isinstance(args[1], np.ndarray):
            d = args[1]
            ppoly = scipy.interpolate.CubicSpline(args[0],  d,
                                                  bc_type="periodic" if is_periodic else "not-a-knot")
        elif isinstance(args[1], scipy.interpolate.PPoly):
            d = args[1](args[0])
            ppoly = args[1]

        if d is None:
            raise NotImplementedError(f"Illegal input {[type(a) for a in args]}")

        obj = Quantity.__new__(cls, d, **kwargs)
        obj._ppoly = ppoly

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._ppoly = getattr(obj, '_ppoly', None)

    def __init__(self,  x, y, *args, is_periodic=False, **kwargs):
        super().__init__(*args,  **kwargs)
        self._is_periodic = is_periodic

    @property
    def is_periodic(self):
        return self._is_periodic

    @property
    def x(self):
        return self._ppoly.x

    @cached_property
    def derivative(self):
        return Function(self.x, self._ppoly.derivative())

    @cached_property
    def antiderivative(self):
        return Function(self.x, self._ppoly.antiderivative())

    @cached_property
    def invert(self):
        try:
            func = Function(self.__call__(self.x), self.x)
        except Exception:
            raise ValueError(f"Can not create invert function!")

        return func

    def integrate(self, a=None, b=None):
        return self._ppoly.integrate(a or self.x[0], b or self.x[-1])

    def __call__(self,   *args, **kwargs):
        return self._ppoly(*args, **kwargs)
