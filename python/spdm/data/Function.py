from __future__ import annotations

import collections.abc
import inspect
import typing
from copy import copy
from functools import cached_property, lru_cache

import numpy as np
from scipy.interpolate import PPoly

from spdm.utils.typing import NumericType

from ..grid.Grid import Grid, as_grid
from ..utils.logger import logger
from ..utils.typing import (ArrayType, NumericType, ScalarType, as_array,
                            as_scalar)
from ..utils.misc import regroup_dict_by_prefix


class Function(object):
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

    TODO:
        - Function[ScalarTypeValue,ScalarTypeGrid] 两个泛型参数，分别描述数值和网格的类型
    """

    def __init__(self, d=None, *args, **kwargs):
        """
        初始化Function 函数
        --------------
        d: np.ndarray | typing.Callable[..., NumericType]
            - 网格点上数值 DoF
            - 描述函数的数值或者插值函数

        """

        if self.__class__ is Function and (kwargs.get("grid_type", "ppoly") == "ppoly"):
            # 若无明确指明 grid_type，默认初始化为 PPolyFunction
            self.__class__ = PPolyFunction
            return self.__class__.__init__(self, d, *args, **kwargs)

        self._data: typing.Any = d  # 网格点上的数值 DoF

        self._grid, self._metadata = regroup_dict_by_prefix(kwargs, "grid")

        if isinstance(self._grid, Grid):
            pass
        elif isinstance(self._grid, collections.abc.Mapping) and len(self._grid) > 0:
            self._grid = as_grid(*args, **self._grid)  # 网格
        elif not self._grid:
            self._grid = None
        else:
            raise RuntimeError(f"Can not convert {type(self._grid)} to Grid!")

        self._ppoly_cache = {}

    @property
    def metadata(self):
        return self._metadata

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}  grid_type=\"{self._grid.name if isinstance(self._grid,Grid) else 'unamed'}\" data_type=\"{self.__type_hint__.__name__}\" />"

    def __serialize__(self) -> typing.Mapping:
        raise NotImplementedError(f"")

    @classmethod
    def __deserialize__(cls, data: typing.Mapping) -> Function:
        raise NotImplementedError(f"")

    def __copy__(self) -> Function:
        """复制 Function """
        other = object.__new__(self.__class__)
        other._grid = self._grid
        other._data = copy(self._data)
        return other

    @property
    def grid(self) -> Grid: return self._grid

    def __array__(self) -> ArrayType:
        """ 重载 numpy 的 __array__ 运算符"""

        if isinstance(self._data, np.ndarray):
            pass
        elif hasattr(self._data, "__value__"):
            self._data = self._data.__value__()
        elif isinstance(self._grid, Grid):
            if isinstance(self._data, (int, float, complex)):
                self._data = np.full(self._grid.shape, self._data)
            else:
                self._data = as_array(self.__call__(self._grid.points))
        else:
            raise RuntimeError(f"Can not convert Function to numpy.ndarray! grid_type={type(self._grid)}")

        return self._data

    def __getitem__(self, *args) -> NumericType:
        return self.__array__().__getitem__(*args)

    def __setitem__(self, *args) -> None:
        return self.__array__().__setitem__(*args)

    # fmt:off
    def __array_ufunc__(self, ufunc, method, *args, **kwargs) -> Expression: return Expression(*args, ufunc=ufunc, method=method, **kwargs)
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

    def __ppoly__(self, *dx: int) -> typing.Callable[..., NumericType]:
        """ 返回 PPoly 对象 """
        fun = self._ppoly_cache.get(dx, None)
        if fun is not None:
            return fun

        if len(dx) == 0:
            if callable(self._data):
                fun = self._data
            elif isinstance(self._grid, Grid):
                fun = self._grid.interpolator(self.__array__())
        else:
            ppoly = self.__ppoly__()

            if all(d < 0 for d in dx):
                if hasattr(ppoly.__class__, 'antiderivative'):
                    fun = self.__ppoly__().antiderivative(*dx)
                elif isinstance(self._grid, Grid):
                    fun = self._grid.antiderivative(self.__array__(), *dx)
            elif all(d >= 0 for d in dx):
                if hasattr(ppoly.__class__, 'partial_derivative'):
                    fun = self.__ppoly__().partial_derivative(*dx)
                elif isinstance(self._grid, Grid):
                    fun = self._grid.partial_derivative(self.__array__(), *dx)

        if fun is None:
            raise RuntimeError(f"Can not convert Function to PPoly! grid_type={type(self._grid)}")

        self._ppoly_cache[dx] = fun

        return fun

    def __call__(self, *args, ** kwargs) -> NumericType: return self.__ppoly__()(*args, ** kwargs)

    def partial_derivative(self, *dx) -> Function: return Function(self.__ppoly__(*dx), self._grid)

    def pd(self, *dx) -> Function: return self.partial_derivative(*dx)

    def antiderivative(self, *dx) -> Function: return Function(self.__ppoly__(*dx), self._grid)

    def integral(self, *dx) -> Function: return self.antiderivative(*dx)

    def dln(self, *dx) -> Function:
        # v = self._interpolator(self._grid)
        # x = (self._grid[:-1]+self._grid[1:])*0.5
        # return Function(x, (v[1:]-v[:-1]) / (v[1:]+v[:-1]) / (self._grid[1:]-self._grid[:-1])*2.0)
        return self.pd(*dx) / self

    def integrate(self, *args, **kwargs) -> ScalarType:
        return as_scalar(self._grid.integrate(self._data, *args, **kwargs))



    # fmt: off
    def __neg__      (self                           ) : return Expression((self,)   ,self._grid, ufunc=np.negative     )
    def __add__      (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.add          )
    def __sub__      (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.subtract     )
    def __mul__      (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.multiply     )
    def __matmul__   (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.matmul       )
    def __truediv__  (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.true_divide  )
    def __pow__      (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.power        )
    def __eq__       (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.equal        )
    def __ne__       (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.not_equal    )
    def __lt__       (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.less         )
    def __le__       (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.less_equal   )
    def __gt__       (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.greater_equal)
    def __ge__       (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.greater_equal)
    def __radd__     (self, o: NumericType | Function) : return Expression((o, self) ,self._grid, ufunc=np.add          )
    def __rsub__     (self, o: NumericType | Function) : return Expression((o, self) ,self._grid, ufunc=np.subtract     )
    def __rmul__     (self, o: NumericType | Function) : return Expression((o, self) ,self._grid, ufunc=np.multiply     )
    def __rmatmul__  (self, o: NumericType | Function) : return Expression((o, self) ,self._grid, ufunc=np.matmul       )
    def __rtruediv__ (self, o: NumericType | Function) : return Expression((o, self) ,self._grid, ufunc=np.divide       )
    def __rpow__     (self, o: NumericType | Function) : return Expression((o, self) ,self._grid, ufunc=np.power        )
    def __abs__      (self                           ) : return Expression((self,)   ,self._grid, ufunc=np.abs          )
    def __pos__      (self                           ) : return Expression((self,)   ,self._grid, ufunc=np.positive     )
    def __invert__   (self                           ) : return Expression((self,)   ,self._grid, ufunc=np.invert       )
    def __and__      (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.bitwise_and  )
    def __or__       (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.bitwise_or   )
    def __xor__      (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.bitwise_xor  )
    def __rand__     (self, o: NumericType | Function) : return Expression((o, self) ,self._grid, ufunc=np.bitwise_and  )
    def __ror__      (self, o: NumericType | Function) : return Expression((o, self) ,self._grid, ufunc=np.bitwise_or   )
    def __rxor__     (self, o: NumericType | Function) : return Expression((o, self) ,self._grid, ufunc=np.bitwise_xor  )
    def __rshift__   (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.right_shift  )
    def __lshift__   (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.left_shift   )
    def __rrshift__  (self, o: NumericType | Function) : return Expression((o, self) ,self._grid, ufunc=np.right_shift  )
    def __rlshift__  (self, o: NumericType | Function) : return Expression((o, self) ,self._grid, ufunc=np.left_shift   )
    def __mod__      (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.mod          )
    def __rmod__     (self, o: NumericType | Function) : return Expression((o, self) ,self._grid, ufunc=np.mod          )
    def __floordiv__ (self, o: NumericType | Function) : return Expression((self, o) ,self._grid, ufunc=np.floor_divide )
    def __rfloordiv__(self, o: NumericType | Function) : return Expression((o, self) ,self._grid, ufunc=np.floor_divide )
    def __trunc__    (self                           ) : return Expression((self,)   ,self._grid, ufunc=np.trunc        )
    def __round__    (self, n=None                   ) : return Expression((self, n) ,self._grid, ufunc=np.round        )
    def __floor__    (self                           ) : return Expression((self,)   ,self._grid, ufunc=np.floor        )
    def __ceil__     (self                           ) : return Expression((self,)   ,self._grid, ufunc=np.ceil         )
    # fmt: on


class Expression(Function):
    def __init__(self, *args,  ufunc: typing.Callable | None = None, method: str | None = None, **kwargs) -> None:
        if ufunc is None and len(args) > 0 and callable(args[0]):
            ufunc = args[0]
            args = args[1:]
        super().__init__(*args, **kwargs)
        self._ufunc = ufunc
        self._method = method

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}   op=\"{self._ufunc.__name__}\" />"

    def __call__(self,  *args: NumericType, **kwargs) -> ArrayType:
        try:
            dtype = self.__type_hint__
        except TypeError:
            dtype = float

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


def function_like(y: NumericType, *args: NumericType, **kwargs) -> Function:
    if len(args) == 0 and isinstance(y, Function):
        return y
    else:
        return Function(y, *args, **kwargs)


_0 = Expression(lambda *args:   args[0])
_1 = Expression(lambda *args:   args[1])
_2 = Expression(lambda *args:   args[2])
_3 = Expression(lambda *args:   args[3])
_4 = Expression(lambda *args:   args[4])
_5 = Expression(lambda *args:   args[5])
_6 = Expression(lambda *args:   args[6])
_7 = Expression(lambda *args:   args[7])
_8 = Expression(lambda *args:   args[8])
_9 = Expression(lambda *args:   args[9])


class PiecewiseFunction(Function):
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

    def derivative(self, *args, **kwargs) -> NumericType | Function:
        return super().derivative(*args, **kwargs)


class PPolyFunction(Function):
    """ PPolyFunction
        ----------------
        A piecewise polynomial function.
        一维或多维，多项式插值函数，
    """

    def __init__(self, d, *args, **kwargs):
        if isinstance(d, PPoly):
            super().__init__()
            self._ppoly = d
        elif len(args) > 0 and isinstance(args[0], Grid):
            super().__init__(d, args[0])
            if len(kwargs) > 0 or len(args) > 1:
                logger.warning(f"Ignore args {args} and kwargs  {kwargs} ")
        else:
            from ..grid.PPolyGrid import PPolyGrid
            super().__init__(d, PPolyGrid(*args, **kwargs))

    # def __ppoly__(self, *dx) -> PPoly:
    #     """ 获取函数的实际表达式，如插值函数 """
    #     if isinstance(self._data, PPoly):
    #         return self._data
    #     elif not isinstance(self._grid, Grid):
    #         raise RuntimeError(f"grid is not Grid, {self._grid}")
    #     elif len(dx) == 0:
    #         return self._grid.interpolator(as_array(self._data))
    #     else:
    #         ppoly = self.__ppoly__()
    #         if dx[0] > 0:
    #             return ppoly.derivative(dx[0])
    #         elif dx[0] == -1:
    #             return ppoly.antiderivative(dx[0])

    def __call__(self, *args, ** kwargs) -> NumericType:

        res = self.__ppoly__()(*args, **kwargs)

        return as_array(res)

    def derivative(self, *args, **kwargs) -> NumericType | Function:
        if isinstance(self.__ppoly__, PPoly):
            return Function(self.__ppoly__.derivative(*args, **kwargs), self._grid)
        else:
            return super().derivative(*args, **kwargs)

    def antiderivative(self, *args, **kwargs) -> NumericType | Function:
        if isinstance(self.__ppoly__, PPoly):
            return Function(self.__ppoly__.antiderivative(*args, **kwargs))
        else:
            return super().antiderivative(*args, **kwargs)

    def integrate(self, *args, **kwargs) -> ScalarType:
        if self.__ppoly__ is not None:
            return as_scalar(self.__ppoly__.integrate(*args, **kwargs))
        else:
            return super().integrate(*args, **kwargs)

    def dln(self, *args, **kwargs) -> NumericType | Function:
        return self.derivative(*args, **kwargs) / self.__call__(*args, **kwargs)
