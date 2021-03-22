from functools import cached_property

import numpy as np
import scipy.interpolate
from ..util.logger import logger
from .Quantity import Quantity


class Function(Quantity):
    @staticmethod
    def __new__(cls,  *args, **kwargs):
        if cls is not Function:
            return Quantity.__new__(cls, *args, **kwargs)
        obj = None
        if len(args) != 2:
            pass
        elif isinstance(args[1], Function):
            obj = Quantity.__new__(cls, args[1](args[0]), **kwargs)
            obj._spl = args[1]._spl
        elif isinstance(args[1], np.ndarray):
            obj = Quantity.__new__(cls, args[1], **kwargs)
            obj._spl = scipy.interpolate.make_interp_spline(args[0], args[1])
        elif isinstance(args[1], scipy.interpolate.BSpline):
            obj = Quantity.__new__(cls, args[1](args[0]), **kwargs)
            obj._spl = args[1]

        if obj is None:
            raise NotImplementedError(f"Illegal input {[type(a) for a in args]}")

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._spl = getattr(obj, '_spl', None)

    def __init__(self,  x, y, *args, is_period=False, **kwargs):
        super().__init__(*args,  **kwargs)
        self._x = x
        self._is_period = is_period
        if self._is_period:
            self._spl.extrapolate = 'periodic'

    @property
    def is_period(self):
        return self._is_period

    @cached_property
    def derivative(self):
        return Function(self._x, self._spl.derivative())

    @cached_property
    def antiderivative(self):
        return Function(self._x, self._spl.antiderivative())

    @cached_property
    def invert(self):
        try:
            func = Function(self.__call__(self._x), self._x)
        except Exception:
            raise ValueError(f"Can not create invert function!")

        return func

    def integrate(self, a=None, b=None):
        return self._spl.integrate(a or self._x[0], b or self._x[-1])

    def __call__(self, *args, **kwargs):
        return self._spl(*args, **kwargs)
