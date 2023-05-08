from __future__ import annotations

import collections.abc
import inspect
import typing
from copy import copy, deepcopy
from functools import cached_property

import numpy as np
from scipy.interpolate import NdPPoly, PPoly

from ..grid.Grid import Grid, as_grid
from ..utils.logger import logger
from ..utils.typing import ArrayType, ScalarType, is_arraylike, is_scalarlike, as_array, as_scalar, NumericType

_T = typing.TypeVar("_T")


class Function(typing.Generic[_T]):
    """
    Function 函数
    --------------
    用于描述一个函数（流形上的映射。。。），
    - 可以是一个数值函数，也可以是一个插值函数
    - 可以是一个标量函数，也可以是一个矢量函数
    - 建立在一维或者多维网格上

    _grid: Grid 以网格的形式描述函数所在流形，
        - Grid.points 网格点坐标

    _data: np.ndarray | typing.Callable[..., NumericType]
        - 网格点上数值 DoF
        - 描述函数的数值或者插值函数
    """

    def __init__(self,  *args, **kwargs):
        """
        初始化Function 函数
        --------------

        """
        grid_type = kwargs.get("grid_type", None)

        if self.__class__ is not Function or grid_type is not None:
            pass
        elif len(args) == 0:
            pass
        elif isinstance(args[0], PPoly) or all([is_arraylike(arg) for arg in args]):
            # 若 grid_type 为 None，且 args 中的所有参数都是 np.ndarray 或者 float，则初始化网格为多项式插值函数 PPolyFunction
            self.__class__ = PPolyFunction
            return self.__class__.__init__(self, *args, **kwargs)
        elif callable(args[0]) or kwargs.get('ufunc', None) is not None:
            # 若 args[0] 是一个函数，则初始化为表达式
            self.__class__ = Expression
            return self.__class__.__init__(self, *args, **kwargs)

        self._data: typing.Any = args[0] if len(args) > 0 else None  # type:ignore  网格点上的数值 DoF
        self._grid: Grid = as_grid(*args[1:], **kwargs)             # type:ignore 网格

        # raise RuntimeError(f"Can not determine the type of Function from args={args} and kwargs={kwargs}")
        # elif self.__class__ is Function and grid_type == "picewise":
        #     # 若 grid_type 为 picewise，则初始化为分段函数 PiecewiseFunction
        #     self.__class__ = PiecewiseFunction
        #     return self.__class__.__init__(self, *args, **kwargs)
        # elif self.__class__ is Function and grid_type == "ppoly":
        #     # 若 grid_type 为 picewise，则初始化为分段函数 PiecewiseFunction
        #     self.__class__ = PPolyFunction
        #     return self.__class__.__init__(self, *args, **kwargs)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}  grid_type=\"{self._grid.name if isinstance(self._grid,Grid) else 'unamed'}\" data_type=\"{self.__type_hint__.__name__}\" />"

    def __copy__(self) -> Function[_T]:
        """复制 Function """
        other = object.__new__(self.__class__)
        other._grid = self._grid
        other._data = copy(self._data)
        return other

    @property
    def grid(self) -> Grid: return self._grid

    # fmt:off
    def __array__(self) -> ArrayType: 
        """ 重载 numpy 的 __array__ 运算符"""

        if self._data is not None:
            return as_array(self._data)
        elif isinstance(self._grid, Grid):
            return as_array(self.__call__(self._grid.points))
        else:
            raise RuntimeError(f"Can not convert Function to numpy.ndarray! grid_type={type(self._grid)}")    



    def __array_ufunc__(self, ufunc, method, *args, **kwargs) -> Expression[_T]: return Expression[_T](*args, ufunc=ufunc, method=method, **kwargs)
    """ 重载 numpy 的 ufunc 运算符"""
    # fmt:on

    @cached_property
    def __type_hint__(self) -> typing.Type:
        """ 获取函数的类型
        """
        orig_class = getattr(self, "__orig_class__", None)
        if orig_class is not None:
            return typing.get_args(orig_class)[0]
        elif isinstance(self._data, np.ndarray):
            return self._data.dtype.type
        elif callable(self._data):
            return typing.get_type_hints(self._data).get("return", None)
        else:
            return None

    def __call__(self, *args, ** kwargs) -> NumericType | Function[_T]:
        if not isinstance(self._grid, Grid):
            raise RuntimeError(f"Grid is not defined! {self._grid}")
        res = self._grid.interpolator(as_array(self._data), *args, **kwargs)
        return Function(res, self._grid) if callable(res) else res

    def derivative(self, *args, **kwargs) -> NumericType | Function[_T]:
        if not isinstance(self._grid, Grid):
            raise RuntimeError(f"Grid is not defined! {self._grid}")
        res = self._grid.derivative(self._data, *args, **kwargs)
        return Function(res, self._grid) if callable(res) else res

    def antiderivative(self, *args, **kwargs) -> NumericType | Function[_T]:
        if not isinstance(self._grid, Grid):
            raise RuntimeError(f"Grid is not defined! {self._grid}")
        res = self._grid.antiderivative(self._data, *args, **kwargs)
        return Function(res, self._grid) if callable(res) else res

    def integrate(self, *args, **kwargs) -> ScalarType:
        if not isinstance(self._grid, Grid):
            raise RuntimeError(f"Grid is not defined! {self._grid}")
        return as_scalar(self._grid.integrate(self._data, *args, **kwargs))

    def dln(self, *args, **kwargs) -> NumericType | Function[_T]:
        # v = self._interpolator(self._grid)
        # x = (self._grid[:-1]+self._grid[1:])*0.5
        # return Function(x, (v[1:]-v[:-1]) / (v[1:]+v[:-1]) / (self._grid[1:]-self._grid[:-1])*2.0)
        return self.derivative(*args, **kwargs) / self

    # fmt: off
    def __neg__      (self                               ) -> Expression[_T]: return Expression[_T]((self,)   ,self._grid, ufunc=np.negative     )
    def __add__      (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.add          )
    def __sub__      (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.subtract     )
    def __mul__      (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.multiply     )
    def __matmul__   (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.matmul       )
    def __truediv__  (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.true_divide  )
    def __pow__      (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.power        )
    def __eq__       (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.equal        )
    def __ne__       (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.not_equal    )
    def __lt__       (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.less         )
    def __le__       (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.less_equal   )
    def __gt__       (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.greater_equal)
    def __ge__       (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.greater_equal)
    def __radd__     (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.add          )
    def __rsub__     (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.subtract     )
    def __rmul__     (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.multiply     )
    def __rmatmul__  (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.matmul       )
    def __rtruediv__ (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.divide       )
    def __rpow__     (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.power        )
    def __abs__      (self                               ) -> Expression[_T]: return Expression[_T]((self,)   ,self._grid, ufunc=np.abs          )
    def __pos__      (self                               ) -> Expression[_T]: return Expression[_T]((self,)   ,self._grid, ufunc=np.positive     )
    def __invert__   (self                               ) -> Expression[_T]: return Expression[_T]((self,)   ,self._grid, ufunc=np.invert       )
    def __and__      (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.bitwise_and  )
    def __or__       (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.bitwise_or   )
    def __xor__      (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.bitwise_xor  )
    def __rand__     (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.bitwise_and  )
    def __ror__      (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.bitwise_or   )
    def __rxor__     (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.bitwise_xor  )
    def __rshift__   (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.right_shift  )
    def __lshift__   (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.left_shift   )
    def __rrshift__  (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.right_shift  )
    def __rlshift__  (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.left_shift   )
    def __mod__      (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.mod          )
    def __rmod__     (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.mod          )
    def __floordiv__ (self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((self, o) ,self._grid, ufunc=np.floor_divide )
    def __rfloordiv__(self, o: NumericType | Function[_T]) -> Expression[_T]: return Expression[_T]((o, self) ,self._grid, ufunc=np.floor_divide )
    def __trunc__    (self                               ) -> Expression[_T]: return Expression[_T]((self,)   ,self._grid, ufunc=np.trunc        )
    def __round__    (self, n=None                       ) -> Expression[_T]: return Expression[_T]((self, n) ,self._grid, ufunc=np.round        )
    def __floor__    (self                               ) -> Expression[_T]: return Expression[_T]((self,)   ,self._grid, ufunc=np.floor        )
    def __ceil__     (self                               ) -> Expression[_T]: return Expression[_T]((self,)   ,self._grid, ufunc=np.ceil         )
    # fmt: on


class Expression(Function[_T]):
    def __init__(self, *args,  ufunc: typing.Callable | None, method: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ufunc = ufunc
        self._method = method

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}   op=\"{self._ufunc.__name__}\" />"

    def __call__(self,  *args: ScalarType | ArrayType, **kwargs) -> ArrayType:

        dtype = float if self.__type_hint__ is None else self.__type_hint__

        if not inspect.isclass(dtype):
            dtype = float

        d = [as_array(d(*args, **kwargs) if callable(d) else d, dtype=dtype) for d in self._data]

        if self._method is not None:
            ufunc = getattr(self._ufunc, self._method, None)
            if ufunc is None:
                raise AttributeError(f"{self._ufunc.__class__.__name__} has not method {self._method}!")
            return ufunc(self, *d)
        elif callable(self._ufunc):
            return self._ufunc(*d)  # type: ignore
        else:
            raise ValueError(f"ufunc is not callable ufunc={self._ufunc} method={self._method}")


def function_like(y: _T, *args: ScalarType | ArrayType, **kwargs) -> Function[_T]:
    if len(args) == 0 and isinstance(y, Function):
        return y
    else:
        return Function[_T](y, *args, **kwargs)


class PiecewiseFunction(Function[_T]):
    """ PiecewiseFunction
        ----------------
        A piecewise function. 一维或多维，分段函数
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if not all(isinstance(arg, collections.abc.Sequence) for arg in args):
            raise TypeError(f"PiecewiseFunction only support sequence of (cond, fun), {args}")
        self._fun_list = zip(*args)

    def __call__(self, x) -> NumericType:
        if isinstance(x, float):
            res = [fun(x) for fun, cond in self._fun_list if cond(x)]
            return res[0]
        elif isinstance(x, np.ndarray) and x.ndim == 1:
            res = np.hstack([fun(x[cond(x)]) for fun, cond in self._fun_list])
            if len(res) != len(x):
                raise RuntimeError(f"PiecewiseFunction result length not equal to input length, {len(res)}!={len(x)}")
            return res
        else:
            raise TypeError(f"PiecewiseFunction only support single float or  1D array, {x}")

    def derivative(self, *args, **kwargs) -> NumericType | Function[_T]:
        return super().derivative(*args, **kwargs)


class PPolyFunction(Function[_T]):
    """ PPolyFunction
        ----------------
        A piecewise polynomial function.
        一维或多维，多项式插值函数，
    """

    def __init__(self, *args, **kwargs):
        if len(args) != 2:
            raise RuntimeError(f"args length must be 2, but {len(args)}")
        from ..grid.PPolyGrid import PPolyGrid
        super().__init__(args[0], PPolyGrid(*args[1:], **kwargs))

        if isinstance(self._data, PPoly):
            self._ppoly = self._data
        else:
            self._ppoly = None

    @ property
    def __ppoly__(self) -> PPoly:
        """ 获取函数的实际表达式，如插值函数 """
        if isinstance(self._ppoly, PPoly):
            pass
        elif isinstance(self._grid, Grid):
            self._ppoly = self._grid.interpolator(self._data)
        else:
            raise RuntimeError(f"Can not get the ppoly of {self}")
        return self._ppoly  # type: ignore

    def __call__(self, *args, ** kwargs) -> NumericType:
        return as_array(self.__ppoly__(*args, **kwargs))

    def derivative(self, *args, **kwargs) -> NumericType | Function[_T]:
        if isinstance(self.__ppoly__, PPoly):
            return Function(self.__ppoly__.derivative(*args, **kwargs), self._grid)
        else:
            return super().derivative(*args, **kwargs)

    def antiderivative(self, *args, **kwargs) -> NumericType | Function[_T]:
        if isinstance(self.__ppoly__, PPoly):
            return Function(self.__ppoly__.antiderivative(*args, **kwargs))
        else:
            return super().antiderivative(*args, **kwargs)

    def integrate(self, *args, **kwargs) -> ScalarType:
        if self.__ppoly__ is not None:
            return as_scalar(self.__ppoly__.integrate(*args, **kwargs))
        else:
            return super().integrate(*args, **kwargs)

    def dln(self, *args, **kwargs) -> NumericType | Function[_T]:
        return self.derivative(*args, **kwargs) / self.__call__(*args, **kwargs)
