from __future__ import annotations

import typing
from copy import copy
import numpy as np

from ..utils.logger import logger
from ..utils.typing import NumericType, numeric_type

ExprOpLike = typing.Callable | None


class Functor:
    """
        算符: 用于表示一个运算符，可以是函数，也可以是类的成员函数
        受 np.ufunc 启发而来。
        可以通过 ExprOp(op, method=method) 的方式构建一个 ExprOp 对象。
    """
    """
    ExprNode
    ---------
    ExprNode= Function + Node 是具有 Function 特性的 Node。
    Function 的 value 由 Node.__value__ 提供，

    mesh 可以根据 kwargs 中的 coordinate* 从 Node 所在 Tree 获得

    """

    def __init__(self, func: typing.Callable | None, /,
                 method: str | None = None,
                 label: str = None,
                 **kwargs) -> None:
        self._func = func
        self._method = method
        self._label = label or func.__class__.__name__
        self._metadata = kwargs

    def __copy__(self) -> Functor:
        """ 复制一个新的 Function 对象 """
        other: Functor = object.__new__(self.__class__)  # type:ignore
        other.__copy_from_(self)
        return other

    def __copy_from_(self, other: Functor) -> Functor:
        """ 复制一个新的 Function 对象 """
        self._func = other._func
        self._method = other._method
        self._metadata = copy(other._metadata)
        return self

    @property
    def __label__(self) -> str: return self._label

    @property
    def __annotation__(self) -> str:
        units = self._metadata.get("units", "")
        label = self._metadata.get("label", None)
        if label is not None:
            return f"{label}  {units}"
        else:
            return units

    def __domain__(self, *args, **kwargs): return True

    def __str__(self) -> str: return self.__label__

    def __call__(self, *args, **kwargs):
        if isinstance(self._method, str):
            op = getattr(self._func, self._method, None)
        elif self._method is not None:
            raise ValueError(self._method)
        else:
            op = self._func

        if not callable(op):
            return op

        try:
            kwargs.update(self._metadata)
            value = op(*args, **kwargs)
        except Exception as error:
            raise RuntimeError(
                f"Error when apply  op={op} args={len(args)} kwargs={kwargs}!") from error

        return value

    def derivative(self, n=1) -> Functor: raise NotImplementedError()

    def partial_derivative(self, *d) -> Functor: raise NotImplementedError()

    def antiderivative(self, *d) -> Functor: raise NotImplementedError()


def as_functor(expr, *args, **kwargs) -> Functor | None:
    if isinstance(expr, Functor):
        return expr
    elif callable(expr):
        return Functor(expr, *args, **kwargs)
    elif expr is None:
        return None
    else:
        raise TypeError(f"expr={expr} is not callable!")
