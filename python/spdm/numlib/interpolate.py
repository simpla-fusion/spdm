
from scipy.interpolate import (InterpolatedUnivariateSpline,
                               RectBivariateSpline, RegularGridInterpolator,
                               UnivariateSpline, interp1d, interp2d)
import collections.abc
import numpy as np
from ..data.Functor import Functor
from ..utils.typing import array_type
from ..utils.logger import logger
import typing


class RectInterpolateOp(Functor):
    def __init__(self, value, *xargs,
                 dims=None, periods=None,
                 check_nan=True,  extrapolate=0,
                 **kwargs) -> None:
        super().__init__(None)
        self._value = value
        self._xargs = xargs
        self._dims = dims
        self._periods = periods
        self._opts = kwargs
        self._check_nan = check_nan
        self._extrapolate = extrapolate
        if isinstance(value, array_type):
            self._shape = tuple(len(d) for d in self._dims) if self._dims is not None else tuple()
            if len(value.shape) > len(self._shape):
                raise NotImplementedError(
                    f"TODO: interpolate for rank >1 . {value.shape}!={self._shape}!  func={self.__str__()} ")
            elif tuple(value.shape) != tuple(self._shape):
                raise RuntimeError(
                    f"Function.compile() incorrect value shape {value.shape}!={self._shape}! func={self.__str__()} ")

    def __call__(self, *args, **kwargs) -> typing.Any:
        if self._func is not None:
            return super().__call__(*args, **kwargs)

        value = self._value

        if callable(value):
            value = value(*np.meshgrid(*self._dims, indexing='ij'))

        if not isinstance(value, array_type):
            raise TypeError((value))

        if len(self._dims) == 1:
            x = self._dims[0]
            if self._check_nan:
                mark = np.isnan(value)
                nan_count = np.count_nonzero(mark)
                if nan_count > 0:
                    # logger.warning(
                    #     f"{self.__class__.__name__}[{self.__str__()}]: Ignore {nan_count} NaN at {np.argwhere(mark)}.")
                    value = value[~mark]
                    x = x[~mark]

            self._func = InterpolatedUnivariateSpline(x, value,  ext=self._extrapolate)
        elif len(self._dims) == 2 and all(d.ndim == 1 for d in self._dims):
            if self._check_nan:
                mark = np.isnan(value)
                nan_count = np.count_nonzero(mark)
                if nan_count > 0:
                    logger.warning(
                        f"{self.__class__.__name__}[{self.__str__()}]: Replace  {nan_count} NaN by 0 at {np.argwhere(mark)}.")
                    value[mark] = 0.0

            # if isinstance(self.periods, collections.abc.Sequence):
            #     logger.warning(f"TODO: periods={self.periods}")

            self._func = RectBivariateSpline(*self._dims, value, **self._opts)
            self._opts = {"grid": False}
        elif all(d.ndim == 1 for d in self._dims):
            self._func = RegularGridInterpolator(self._dims, value, **self._opts)
        else:
            raise NotImplementedError(f"dims={self._dims} ")

        return super().__call__(*args, **kwargs)

    def derivative(self, n=1) -> Functor:
        fname = f"d_{n}({self.__str__()})"
        if isinstance(n, collections.abc.Sequence) and len(n) == 1:
            n = n[0]
        return Functor(self.__op__.derivative(n),  **self._opts)

    def partial_derivative(self, *d) -> Functor:
        if len(d) > 0:
            fname = f"d_({self.__str__()})"
        else:
            fname = f"d_{d}({self.__str__()})"

        return Functor(self.__op__.partial_derivative(*d), **self._opts)

    def antiderivative(self, *d) -> Functor:
        return Functor(self.__op__.antiderivative(*d), **self._opts)


def interpolate(*args, type="rectlinear", **kwargs) -> Functor:
    if type != "rectlinear":
        raise NotImplementedError(f"type={type}")
    return RectInterpolateOp(*args, **kwargs)
