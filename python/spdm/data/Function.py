from functools import cached_property

import numpy as np
from scipy import constants
import scipy.interpolate
from scipy.interpolate.interpolate import PPoly

from ..util.logger import logger
from .Quantity import Quantity

# if version.parse(scipy.__version__) <= version.parse("1.4.1"):
#     from scipy.integrate import cumtrapz as cumtrapz
# else:
#     from scipy.integrate import cumulative_trapezoid as cumtrapz

logger.debug(f"Using SciPy Version: {scipy.__version__}")


class Function(np.ndarray):
    @staticmethod
    def __new__(cls,  *args, is_periodic=False, **kwargs):
        if cls is not Function:
            return object.__new__(cls, *args, **kwargs)
        ppoly = None
        x = None
        y = None
        if len(args) == 1:
            if isinstance(args[0], Function):
                ppoly = args[0]._ppoly
                x = args[0].x
                y = args[0].view(np.ndarray)
            elif isinstance(args[0], PPoly):
                ppoly = args[0]
                x = ppoly.x
                y = ppoly(x)
            elif isinstance(args[0], np.ndarray):
                x = np.linspace(0, 1.0, len(args[0]))
                y = args[0]
        elif len(args) == 2:
            x = args[0]
            y = args[1]
        else:
            raise NotImplementedError(f"Illegal input {[type(a) for a in args]}")

        if ppoly is not None:
            pass
        elif isinstance(y, scipy.interpolate.PPoly):
            ppoly = y
            y = ppoly(x)
        elif isinstance(y, Function):
            if x is not y.x and any(x != y.x):
                y = y(x)
        elif callable(y):
            y = y(x)
            ppoly = y
        elif isinstance(y, np.ndarray):
            pass
        elif isinstance(y, (float, int)):
            y = np.full(len(x), y)
        else:
            raise NotImplementedError(f"Illegal input {[type(a) for a in args]}")
        if isinstance(y, np.ndarray):
            y = y.view(cls)

        y._ppoly = ppoly
        y._x = x
        y._is_periodic = is_periodic
        return y

    def __array_finalize__(self, obj):
        self._ppoly = getattr(obj, '_ppoly', None)
        self._x = getattr(obj, '_x', None)
        self._is_periodic = getattr(obj, '_is_periodic', None)

    def __init__(self,  x, y=None, *args, **kwargs):
        # super().__init__(*args,  **kwargs)
        pass

    @property
    def ppoly(self) -> PPoly:
        if self._ppoly is None:
            self._ppoly = scipy.interpolate.CubicSpline(
                self.x, self.view(np.ndarray), bc_type="periodic" if self.is_periodic else "not-a-knot")
        return self._ppoly

    @property
    def is_periodic(self):
        return self._is_periodic

    @property
    def x(self):
        return self._x

    @cached_property
    def derivative(self):
        return Function(self.ppoly.derivative())

    @cached_property
    def antiderivative(self):
        return Function(self.ppoly.antiderivative())

    @cached_property
    def invert(self):
        try:
            func = Function(self.__call__(self.x), self.x)
        except Exception:
            raise ValueError(f"Can not create invert function!")

        return func

    def integrate(self, a=None, b=None):
        return self.ppoly.integrate(a or self.x[0], b or self.x[-1])

    def __call__(self,   *args, **kwargs):
        return self.ppoly(*args, **kwargs)

    def pullback(self, *args):
        if len(args) == 0:
            raise ValueError(f"missing arguments!")
        elif len(args) == 2 and args[0].shape == args[1].shape:
            x0, x1 = args
            y = self(x0)
        elif isinstance(args[0], Function) or callable(args[0]):
            x1 = args[0](self.x)
            y = self.view(np.ndarray)
        elif isinstance(args[0], np.ndarray):
            x1 = args[0]
            y = self(x1)
        else:
            raise TypeError(f"{args}")

        return Function(x1, y, is_periodic=self.is_periodic)
