from functools import cached_property

import numpy as np
import scipy.interpolate
from scipy.interpolate.interpolate import PPoly
from typing import Any
from ..util.logger import logger

# from ..data.Quantity import Quantity

# if version.parse(scipy.__version__) <= version.parse("1.4.1"):
#     from scipy.integrate import cumtrapz as cumtrapz
# else:
#     from scipy.integrate import cumulative_trapezoid as cumtrapz

logger.debug(f"SciPy: Version {scipy.__version__}")


class PimplFunc(object):
    def __init__(self,  *args,   ** kwargs) -> None:
        super().__init__()

    @property
    def is_periodic(self):
        return False

    @property
    def x(self) -> np.ndarray:
        return NotImplemented

    @property
    def y(self) -> np.ndarray:
        return self.apply(self.x)

    def apply(self, x) -> np.ndarray:
        raise NotImplementedError(self.__class__.__name__)

    @cached_property
    def derivative(self):
        return SplineFunction(self.x, self.y, is_periodic=self.is_periodic).derivative

    @cached_property
    def antiderivative(self):
        return SplineFunction(self.x, self.y, is_periodic=self.is_periodic).antiderivative

    def invert(self, x=None):
        x = self.x if x is None else x
        return PimplFunc(self.apply(x), x)


class WrapperFunc(PimplFunc):
    def __init__(self, x, func, * args,   ** kwargs) -> None:
        super().__init__()
        self._x = x
        self._func = func

    def apply(self, x) -> np.ndarray:
        x = self.x if x is None else x
        return self._func(x)

    @property
    def x(self) -> np.ndarray:
        return self._x


class SplineFunction(PimplFunc):
    def __init__(self, x, y=None, is_periodic=False,  ** kwargs) -> None:
        super().__init__()
        self._is_periodic = is_periodic

        if isinstance(x, PPoly) and y is None:
            self._ppoly = x
        elif not isinstance(x, np.ndarray) or y is None:
            raise TypeError((type(x), type(y)))
        else:
            if callable(y):
                y = y(x)
            elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                assert(x.shape == y.shape)
            elif isinstance(y, (float, int)):
                y = np.full(len(x), y)
            else:
                raise NotImplementedError(f"Illegal input {[type(a) for a in args]}")

            if is_periodic:
                self._ppoly = scipy.interpolate.CubicSpline(x, y, bc_type="periodic", **kwargs)
            else:
                self._ppoly = scipy.interpolate.CubicSpline(x, y, **kwargs)

    @property
    def x(self) -> np.ndarray:
        return self._ppoly.x

    @cached_property
    def derivative(self):
        return SplineFunction(self._ppoly.derivative())

    @cached_property
    def antiderivative(self):
        return SplineFunction(self._ppoly.antiderivative())

    def apply(self, x) -> np.ndarray:
        x = self.x if x is None else x
        return self._ppoly(x)

    def integrate(self, a, b):
        return self._ppoly.integrate(a, b)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.apply(*args, **kwds)


class PiecewiseFunction(PimplFunc):
    def __init__(self, x, cond, func, *args,    **kwargs) -> None:
        super().__init__()
        self._x = x
        self._cond = cond
        self._func = func

    @property
    def x(self) -> np.ndarray:
        return self._x

    def apply(self, x) -> np.ndarray:
        cond = [c(x) for c in self._cond]
        return np.piecewise(x, cond, self._func)


class Expression(PimplFunc):
    def __init__(self, ufunc, method, *inputs,  **kwargs) -> None:
        super().__init__()
        self._ufunc = ufunc
        self._method = method
        self._inputs = inputs
        self._kwargs = kwargs

    @property
    def is_periodic(self):
        return all([d.is_periodic for d in self._inputs if isinstance(d, Function)])

    @property
    def x(self):
        return next(d.x for d in self._inputs if isinstance(d, Function))

    @property
    def y(self) -> np.ndarray:
        return self.apply(self.x)

    def apply(self, x) -> np.ndarray:
        def wrap(x, d):
            if isinstance(d, Function):
                res = d(x).view(np.ndarray)
            elif not isinstance(d, np.ndarray):
                res = d
            elif d.shape == self.x.shape:
                res = Function(self.x, d)(x).view(np.ndarray)
            else:
                raise ValueError(f"{self.x.shape} {d.shape}")

            return res

        if self._method != "__call__":
            op = getattr(self._ufunc, self._method)
            # raise RuntimeError((self._ufunc, self._method))
            res = op(*[wrap(x, d) for d in self._inputs])
        try:
            res = self._ufunc(*[wrap(x, d) for d in self._inputs])
        except Warning as error:
            raise ValueError(
                f"\n {self._ufunc}  {[type(a) for a in self._inputs]}  {[a.shape for a in self._inputs if isinstance(a,Function)]} {error} \n ")
        return res


class Function:

    def __init__(self, x, y=None, *args,  is_periodic=False,  experiment=False, **kwargs):
        self._is_periodic = is_periodic

        self._pimpl = None
        self._x = x

        if y is None:
            if isinstance(x, Function):
                pimpl = x._pimpl
                self._x = x._x
            elif isinstance(x, PimplFunc):
                self._pimpl = x
                self._x = self._pimpl.x
            elif isinstance(x, np.ndarray):
                self._x = np.linspace(0, 1.0, x.shape)
                self._pimpl = SplineFunction(self._x, x, is_periodic=is_periodic)
        elif not isinstance(x, np.ndarray):
            raise TypeError(f"x should be np.ndarray not {type(x)}!")
        elif isinstance(y, Function):
            self._pimpl = y._pimpl
        elif isinstance(y, PimplFunc):
            self._pimpl = y
        elif isinstance(y, np.ndarray):
            if(getattr(self._x, "shape", None) != getattr(y, "shape", None)):
                raise RuntimeError((getattr(self._x, "shape", None), getattr(y, "shape", None)))
            self._pimpl = SplineFunction(self._x, y, is_periodic=self.is_periodic)
        elif isinstance(y, (int, float, complex)) or y is None or y == None or callable(y):
            self._pimpl = SplineFunction(self._x, y, is_periodic=self.is_periodic)
        else:
            raise TypeError(type(y))

    def __array__(self) -> np.ndarray:
        if not hasattr(self, "_cache"):
            self._cache = self._pimpl(self._x)
        return self._cache

    def __array_ufunc__(self, ufunc, method, *inputs,   **kwargs):
        logger.debug(ufunc)
        return Function(Expression(ufunc, method, *inputs, **kwargs))

    def __repr__(self) -> str:
        return self.__array__().__repr__()

    def __getitem__(self, key):
        # d = super().__getitem__(key)
        # if isinstance(d, np.ndarray) and len(d.shape) > 0:
        #     d = d.view(Function)
        #     d._pimpl = self._pimpl
        #     d._x = self.x[key]
        return self.__array__()[key]

    def __setitem__(self, idx, value):
        raise NotImplementedError

    @cached_property
    def spl(self):
        if isinstance(self._pimpl, SplineFunction):
            return self._pimpl
        else:
            return SplineFunction(self.x, self.__array__(), is_periodic=self.is_periodic)

    @property
    def is_periodic(self):
        return self._is_periodic

    @property
    def x(self):
        return self._x

    @cached_property
    def derivative(self):
        return Function(self.spl.derivative)

    @cached_property
    def antiderivative(self):
        return Function(self.spl.antiderivative)

    @cached_property
    def invert(self):
        return Function(self.spl.invert(self._x))

    def pullback(self, *args, **kwargs):
        if len(args) == 0:
            raise ValueError(f"missing arguments!")
        elif len(args) == 2 and args[0].shape == args[1].shape:
            x0, x1 = args
            y = self(x0)
        elif isinstance(args[0], Function) or callable(args[0]):
            logger.warning(f"FIXME: not complete")
            x1 = args[0](self.x)
            y = self.view(np.ndarray)
        elif isinstance(args[0], np.ndarray):
            x1 = args[0]
            y = self(x1)
        else:
            raise TypeError(f"{args}")

        return Function(x1, y, is_periodic=self.is_periodic)

    def integrate(self, a=None, b=None):
        return self.spl.integrate(a or self.x[0], b or self.x[-1])

    def __call__(self,   *args, **kwargs):
        if len(args) == 0:
            args = [self._x]
        if hasattr(self._pimpl, "apply"):
            res = self._pimpl.apply(*args, **kwargs)
        elif callable(self._pimpl):
            res = self._pimpl(*args, **kwargs)
        else:
            res = self.spl.apply(*args, **kwargs)
            # raise RuntimeError(f"{type(self.spl)}")

        return res.view(np.ndarray)
