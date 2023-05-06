from __future__ import annotations

import collections.abc
import inspect
import typing
import warnings
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..grid.Grid import Grid, NullGrid
from ..grid.PPolyGrid import PPolyGrid
from ..utils.logger import logger
from ..utils.misc import float_unique
from ..utils.tags import _not_found_

_T = typing.TypeVar("_T")


class Function(typing.Generic[_T]):
    """
        NOTE: Function is immutable!!!!
    """

    def __init__(self,  *args, **kwargs):
        if self.__class__ is Function and (len(args) > 0 and callable(args[0])) or kwargs.get('ufunc', None) is not None:
            self.__class__ = Expression
            return self.__class__.__init__(self, *args, **kwargs)
        elif len(args) == 0:
            self._data = tuple()
            self._grid = None
        else:
            self._data = args[0]
            if len(args) == 2 and isinstance(args[1], Grid):
                self._grid = args[1]
                self._appinfo = kwargs
            elif all([isinstance(a, np.ndarray) for a in args[1:]]):
                self._grid = PPolyGrid(*args[1:], **kwargs)
            else:
                self._grid = Grid(*args[1:], **kwargs)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}  grid_type=\"{self._grid.name}\" data_type=\"{self.__type_hint__.__name__}\" />"

    def __duplicate__(self) -> Function[_T]:
        other = object.__new__(self.__class__)
        other._grid = self._grid
        other._data = self._data
        return other

    @property
    def grid(self) -> Grid | None:
        return self._grid

    def __array__(self) -> NDArray: return np.asarray(self.__call__())

    def __array_ufunc__(self, ufunc, method, *args,   **kwargs) -> Expression[_T]:
        return Expression[_T](*args, ufunc=ufunc, method=method, **kwargs)

    @cached_property
    def __type_hint__(self) -> typing.Type:
        orig_class = getattr(self, "__orig_class__", None)
        if orig_class is not None:
            return typing.get_args(orig_class)[0]
        elif isinstance(self._data, np.ndarray):
            return self._data.dtype.type
        elif callable(self._data):
            return typing.get_type_hints(self._data).get("return", None)
        else:
            return None

    @cached_property
    def __fun__(self) -> typing.Callable[..., ArrayLike | NDArray]:
        if self._grid is not None:
            return self._grid.interpolator(self._data)
        elif isinstance(self._data, (int, float, bool, complex)):
            v = self._data
            return lambda *_: self.__type_hint__(v)
        else:
            raise RuntimeError(f"Function is not callable!")

    def __call__(self, *args, **kwargs) -> ArrayLike | NDArray: return self.__fun__(*args, **kwargs)

    def derivative(self, *args, **kwargs) -> Function[_T]:
        if self.__fun__ is not None:
            return Function(self.__fun__.derivative(*args, **kwargs))
        elif self._grid is not None:
            return Function(self._grid.derivative(self.__array__(), *args, **kwargs))
        else:
            raise RuntimeError(f"Function is not callable! {self.__fop__}")

    def antiderivative(self, *args, **kwargs) -> Function[_T]:
        if self.__fun__ is not None:
            return Function(self.__fun__.antiderivative(*args, **kwargs))
        elif self._grid is not None:
            return Function(self._grid.antiderivative(self.__array__(), *args, **kwargs))
        else:
            raise RuntimeError(f"Function is not callable! {self.__fop__}")

    def integrate(self, *args, **kwargs) -> _T:
        if self.__fun__ is not None:
            return self.__fun__.integrate(*args, **kwargs)
        elif self._grid is not None:
            return self._grid.integrate(self.__array__(), *args, **kwargs)
        else:
            raise RuntimeError(f"")

    def dln(self, *args, **kwargs) -> Function[_T]:
        # v = self._interpolator(self._grid)
        # x = (self._grid[:-1]+self._grid[1:])*0.5
        # return Function(x, (v[1:]-v[:-1]) / (v[1:]+v[:-1]) / (self._grid[1:]-self._grid[:-1])*2.0)
        return self.derivative(*args, **kwargs) / self

    # fmt: off
    def __neg__      (self                      ) -> Expression[_T]: return Expression[_T]((self,)   ,self._grid, ufunc=np.negative     )
    def __add__      (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.add          )
    def __sub__      (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.subtract     )
    def __mul__      (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.multiply     )
    def __matmul__   (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.matmul       )
    def __truediv__  (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.true_divide  )
    def __pow__      (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.power        )
    def __eq__       (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.equal        )
    def __ne__       (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.not_equal    )
    def __lt__       (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.less         )
    def __le__       (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.less_equal   )
    def __gt__       (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.greater_equal)
    def __ge__       (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.greater_equal)
    def __radd__     (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.add          )
    def __rsub__     (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.subtract     )
    def __rmul__     (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.multiply     )
    def __rmatmul__  (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.matmul       )
    def __rtruediv__ (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.divide       )
    def __rpow__     (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.power        )
    def __abs__      (self                      ) -> Expression[_T]: return Expression[_T]((self,)   ,self._grid, ufunc=np.abs          )
    def __pos__      (self                      ) -> Expression[_T]: return Expression[_T]((self,)   ,self._grid, ufunc=np.positive     )
    def __invert__   (self                      ) -> Expression[_T]: return Expression[_T]((self,)   ,self._grid, ufunc=np.invert       )
    def __and__      (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.bitwise_and  )
    def __or__       (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.bitwise_or   )
    def __xor__      (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.bitwise_xor  )
    def __rand__     (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.bitwise_and  )
    def __ror__      (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.bitwise_or   )
    def __rxor__     (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.bitwise_xor  )
    def __rshift__   (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.right_shift  )
    def __lshift__   (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.left_shift   )
    def __rrshift__  (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.right_shift  )
    def __rlshift__  (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.left_shift   )
    def __mod__      (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.mod          )
    def __rmod__     (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.mod          )
    def __floordiv__ (self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.floor_divide )
    def __rfloordiv__(self, o: ArrayLike|NDArray) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.floor_divide )
    def __trunc__    (self                      ) -> Expression[_T]: return Expression[_T]((self,)   ,self._grid, ufunc=np.trunc        )
    def __round__    (self, n=None              ) -> Expression[_T]: return Expression[_T]((self, n) ,self._grid, ufunc=np.round        )
    def __floor__    (self                      ) -> Expression[_T]: return Expression[_T]((self,)   ,self._grid, ufunc=np.floor        )
    def __ceil__     (self                      ) -> Expression[_T]: return Expression[_T]((self,)   ,self._grid, ufunc=np.ceil         )
    # fmt: on


class PicewiseFunction(Function[_T]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Expression(Function[_T]):
    def __init__(self, *args,  ufunc: typing.Callable | None, method: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ufunc = ufunc
        self._method = method

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}   op=\"{self._ufunc.__name__}\" />"

    def __call__(self,  *args: ArrayLike | NDArray | Grid, **kwargs) -> ArrayLike | NDArray:

        dtype = float if self.__type_hint__ is None else self.__type_hint__

        if not inspect.isclass(dtype):
            dtype = float

        d = [np.asarray(d(*args)if callable(d) else d, dtype=dtype) for d in self._data]

        if self._method is not None:
            ufunc = getattr(self._ufunc, self._method)
            return ufunc(self, *d)
        elif callable(self._ufunc):
            return ufunc(*d)  # type: ignore
        else:
            raise ValueError(f"ufunc is not callable ufunc={self._ufunc} method={self._method}")


def function_like(y: _T, *args: ArrayLike, **kwargs) -> Function[_T]:
    if len(args) == 0 and isinstance(y, Function):
        return y
    else:
        return Function[_T](y, *args, **kwargs)
