from __future__ import annotations

import collections.abc
import typing
from enum import Enum
from functools import cached_property
from copy import copy
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline

from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix
from ..utils.tags import _not_found_
from ..utils.typing import (ArrayType, NumericType, ScalarType, as_scalar,
                            numeric_types)
from .Expression import Expression

_T = typing.TypeVar("_T")


class Function(Expression[_T]):
    """
        Function
        ---------
        A function is a mapping between two sets, the _domain_ and the  _value_.
        The _value_  is the set of all possible outputs of the function.
        The _domain_ is the set of all possible inputs  to the function.
    """

    def __init__(self, value: NumericType, *domain: ArrayType, op=None, cycles=None, **kwargs):
        """
            Parameters
            ----------
            value : NumericType
                函数的值
            domain : typing.List[ArrayType]
                函数的定义域
            kwargs : typing.Any
                命名参数， 用于传递给运算符的参数

        """
        if callable(value):
            if op is not None:
                raise RuntimeError(f"Can not specify both value and op!")
            op = value
            value = None

        super().__init__(op=op, **kwargs)

        self._value = value

        self._domain = domain

        if not all(isinstance(d, np.ndarray) for d in self._domain):
            raise RuntimeError(f"Function domain must be all np.ndarray!")

        self._cycles = cycles if cycles is not None else [np.inf]*len(self._domain)

        self._ppoly_cache = {}

    def __duplicate__(self) -> Function:
        """ 复制一个新的 Function 对象 """
        other: Function = super().__duplicate__()
        other._value = copy(self._value)
        other._domain = self._domain
        other._ppoly_cache = {}
        return other

    def __serialize__(self) -> typing.Mapping: raise NotImplementedError(f"")

    @ classmethod
    def __deserialize__(cls, data: typing.Mapping) -> Function: raise NotImplementedError(f"")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}  mesh_type=\"{self._domain.__class__.__name__}\" data_type=\"{self.__type_hint__.__name__}\" />"

    @ property
    def rank(self) -> int: return len(self._domain)
    """ 函数的秩，即定义域的维度 """

    @ property
    def domain(self): return self._domain

    @ cached_property
    def bbox(self) -> typing.Tuple[ArrayType, ArrayType]:
        """ bound box 返回包裹函数参数的取值范围的最小多维度超矩形（hyperrectangle） """
        return (np.asarray([np.min(d) for d in self._domain], dtype=float), np.asarray([np.max(d) for d in self._domain], dtype=float))

    def __array__(self) -> ArrayType: return self._array

    @ cached_property
    def _array(self) -> ArrayType:
        """ 重载 numpy 的 __array__ 运算符
            若 self._value 为 np.ndarray 或标量类型 则返回 self._data, 否则返回 None
        """
        if self._value is None:
            self._value = super().__call__(*self._domain)
        elif hasattr(self._value, "__entry__"):
            self._value = self._data.__entry__().__value__()

        if not isinstance(self._value, np.ndarray):
            self._value = np.asarray(self._value, dtype=self.__type_hint__)

        return self._value

    def __getitem__(self, *args) -> NumericType: return self._array.__getitem__(*args)

    def __setitem__(self, *args) -> None: raise RuntimeError("Function.__setitem__ is prohibited!")

    @ cached_property
    def __type_hint__(self) -> typing.Type:
        """ 获取函数的类型
        """
        orig_class = getattr(self, "__orig_class__", None)
        if orig_class is not None:
            return typing.get_args(orig_class)[0]
        elif isinstance(self._value, np.ndarray):
            return self._value.dtype.type
        else:
            return float

    def __ppoly__(self, *dx: int) -> typing.Callable[..., NumericType]:
        """ 返回 PPoly 对象
            TODO:
            - support JIT compile
            - 优化缓存
            - 支持多维插值
            - 支持多维求导，自动微分 auto diff
            -
        """

        fun = self._ppoly_cache.get(dx, None)

        if fun is not None:
            return fun

        ppoly = self._ppoly_cache.get((), None)

        if ppoly is None:
            if len(self._domain) == 1 and isinstance(self._domain[0], np.ndarray):
                ppoly = InterpolatedUnivariateSpline(self._domain[0], self._array)
            elif len(self._domain) == 2 and all(isinstance(d, np.ndarray) for d in self._domain):
                ppoly = RectBivariateSpline(*self._domain, self._array)
            else:
                raise RuntimeError(
                    f"Can not convert Function to PPoly! value={type(self._array)} domain={self._domain} ")
            self._ppoly_cache[()] = ppoly

        if len(dx) > 0:

            if all(d < 0 for d in dx):
                ppoly = ppoly.antiderivative(*[-d for d in dx])
            elif all(d >= 0 for d in dx):
                ppoly = ppoly.partial_derivative(*dx)
            else:
                raise RuntimeError(f"{dx}")
            self._ppoly_cache[dx] = ppoly

        return ppoly

    def __call__(self, *args, ** kwargs) -> NumericType:
        if self._op is None:
            self._op = self.__ppoly__()
        return super().__call__(*args, ** kwargs)

    def partial_derivative(self, *dx) -> Function: return Function(self.__ppoly__(dx), self._domain, **self._kwargs)

    def pd(self, *dx) -> Function: return self.partial_derivative(*dx)

    def antiderivative(self, *dx) -> Function: return Function(self.__ppoly__(*dx), self._domain, **self._kwargs)

    def dln(self, *dx) -> Function: return self.pd(*dx) / self
    # v = self._interpolator(self._mesh)
    # x = (self._mesh[:-1]+self._mesh[1:])*0.5
    # return Function(x, (v[1:]-v[:-1]) / (v[1:]+v[:-1]) / (self._mesh[1:]-self._mesh[:-1])*2.0)

    def integral(self, a, b) -> Function: return self.__ppoly__().integral(a, b)

    def roots(self, **kwargs): return self.__ppoly__().roots(**kwargs)


def function_like(y: NumericType, *args: NumericType, **kwargs) -> Function:
    if len(args) == 0 and isinstance(y, Function):
        return y
    else:
        return Function(y, *args, **kwargs)


class Piecewise(Expression[_T]):
    """ PiecewiseFunction
        ----------------
        A piecewise function. 一维或多维，分段函数
    """

    def __init__(self, func: typing.List[typing.Callable], cond: typing.List[typing.Callable], **kwargs):
        super().__init__(op=(func, cond), **kwargs)

    def _apply(self, func, x, *args, **kwargs):
        if isinstance(x, np.ndarray) and isinstance(func, (int, float, complex, np.floating, np.integer, np.complexfloating)):
            value = np.full(x.shape, func)
        else:
            value = super()._apply(func, x, *args, **kwargs)
        return value

    def __call__(self, x, *args, **kwargs) -> NumericType:
        if isinstance(x, float):
            res = [self._apply(fun, x) for fun, cond in zip(*self._op) if cond(x)]
            if len(res) == 0:
                raise RuntimeError(f"Can not fit any condition! {x}")
            elif len(res) > 1:
                raise RuntimeError(f"Fit multiply condition! {x}")
            return res[0]
        elif isinstance(x, np.ndarray):
            res = np.hstack([self._apply(fun, x[cond(x)]) for fun, cond in zip(*self._op)])
            if len(res) != len(x):
                raise RuntimeError(f"PiecewiseFunction result length not equal to input length, {len(res)}!={len(x)}")
            return res
        else:
            raise TypeError(f"PiecewiseFunction only support single float or  1D array, {x}")

    def derivative(self, *args, **kwargs) -> NumericType | Function:
        return super().derivative(*args, **kwargs)
