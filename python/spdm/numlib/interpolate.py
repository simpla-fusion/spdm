
from scipy.interpolate import (InterpolatedUnivariateSpline,
                               RectBivariateSpline, RegularGridInterpolator,
                               UnivariateSpline, interp1d, interp2d)
import collections.abc
import numpy as np
from ..data.ExprOp import ExprOp
from ..utils.typing import array_type
from ..utils.logger import logger
import typing


class RectInterpolateOp(ExprOp):
    def __init__(self, value, *dims, periods=None, check_nan=True, name=None, extrapolate=0, **kwargs) -> None:
        super().__init__(None, name=name)
        self._value = value
        self._dims = dims
        self._periods = periods
        self._opts = kwargs
        self._check_nan = check_nan
        self._extrapolate = extrapolate
        if isinstance(value, array_type):
            self._shape = tuple(len(d) for d in self.dims) if self.dims is not None else tuple()
            if len(value.shape) > len(self._shape):
                raise NotImplementedError(
                    f"TODO: interpolate for rank >1 . {value.shape}!={self._shape}!  func={self.__str__()} ")
            elif tuple(value.shape) != tuple(self._shape):
                raise RuntimeError(
                    f"Function.compile() incorrect value shape {value.shape}!={self._shape}! func={self.__str__()} ")

    @property
    def dims(self) -> typing.Tuple[int]: return self._dims

    @property
    def shape(self) -> typing.Tuple[int]: return self._shape

    @property
    def __op__(self) -> typing.Callable:
        if self._op is not None:
            return self._op

        value = self._value
        m_shape = self.shape

        if callable(value):
            value = value(*np.meshgrid(*self.dims, indexing='ij'))

        if not isinstance(value, array_type):
            raise TypeError(type(value))

        if len(self.dims) == 1:
            x = self.dims[0]
            if self._check_nan:
                mark = np.isnan(value)
                nan_count = np.count_nonzero(mark)
                if nan_count > 0:
                    # logger.warning(
                    #     f"{self.__class__.__name__}[{self.__str__()}]: Ignore {nan_count} NaN at {np.argwhere(mark)}.")
                    value = value[~mark]
                    x = x[~mark]

            self._op = InterpolatedUnivariateSpline(x, value,  ext=self._extrapolate)
        elif len(self.dims) == 2 and all(d.ndim == 1 for d in self.dims):
            if self._check_nan:
                mark = np.isnan(value)
                nan_count = np.count_nonzero(mark)
                if nan_count > 0:
                    logger.warning(
                        f"{self.__class__.__name__}[{self.__str__()}]: Replace  {nan_count} NaN by 0 at {np.argwhere(mark)}.")
                    value[mark] = 0.0

            # if isinstance(self.periods, collections.abc.Sequence):
            #     logger.warning(f"TODO: periods={self.periods}")

            self._op = RectBivariateSpline(*self.dims, value, **self._opts)
            self._opts = {"grid": False}
        elif all(d.ndim == 1 for d in self.dims):
            self._op = RegularGridInterpolator(self.dims, value, **self._opts)
        else:
            raise NotImplementedError(f"dims={self.dims} ")

        return self._op

    def derivative(self, n=1) -> ExprOp:
        fname = f"d_{n}({self.__str__()})"
        if isinstance(n, collections.abc.Sequence) and len(n) == 1:
            n = n[0]
        return ExprOp(self.__op__.derivative(n), name=fname, **self._opts)

    def partial_derivative(self, *d) -> ExprOp:
        if len(d) > 0:
            fname = f"d_({self.__str__()})"
        else:
            fname = f"d_{d}({self.__str__()})"

        return ExprOp(self.__op__.partial_derivative(*d), name=fname, **self._opts)

    def antiderivative(self, *d) -> ExprOp:
        return ExprOp(self.__op__.antiderivative(*d),  name=f"I_{list(d)}({self.__str__()})", **self._opts)


def interpolate(*args, type="rectlinear", **kwargs) -> ExprOp:
    if type != "rectlinear":
        raise NotImplementedError(f"type={type}")
    return RectInterpolateOp(*args, **kwargs)
