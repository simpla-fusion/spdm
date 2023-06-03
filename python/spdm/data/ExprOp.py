from __future__ import annotations

import collections.abc
import functools
import inspect
import typing

import numpy as np

from ..utils.logger import logger
from ..utils.tags import _not_found_, _undefined_
from ..utils.typing import (ArrayType, NumericType, array_type, numeric_type,
                            scalar_type)

_EXPR_OP_NAME = {
    "negative": "-",
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "matmul": "*",
    "true_divide": "/",
    "power": "^",
    "equal": "==",
    "not_equal": "!",
    "less": "<",
    "less_equal": "<=",
    "greater": ">",
    "greater_equal": ">=",
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "matmul": "*",
    "divide": "/",
    "power": "^",
    # "abs": "",
    "positive": "+",
    # "invert": "",
    "bitwise_and": "&",
    "bitwise_or": "|",

    # "bitwise_xor": "",
    # "right_shift": "",
    # "left_shift": "",
    # "right_shift": "",
    # "left_shift": "",
    "mod": "%",
    # "floor_divide": "",
    # "floor_divide": "",
    # "trunc": "",
    # "round": "",
    # "floor": "",
    # "ceil": "",
}


class ExprOp:
    """ 
        算符: 用于表示一个运算符，可以是函数，也可以是类的成员函数
        受 np.ufunc 启发而来。
        可以通过 ExprOp(op, method=method) 的方式构建一个 ExprOp 对象。
    """

    def __init__(self, op, /, method: str = None, name=None, **kwargs) -> None:
        self._op = op
        self._method = method
        self._opts = kwargs

        if name is not None:
            self._name = name
        elif isinstance(op, numeric_type):
            self._name = f"[{type(op)}]"
        elif method is None or method == "__call__":
            self._name = getattr(op, "__name__", None)
        elif method is not None:
            self._name = f"{op.__class__.__name__}.{method}"
        else:
            self._name = ""

    def __str__(self) -> str: return str(self._name)

    @property
    def __name__(self) -> str: return self._name
    """ To get the name of the operator，same as self.name. To compatible with numpy ufunc. """
    @property
    def name(self) -> str: return self._name

    @property
    def tag(self) -> str: return _EXPR_OP_NAME.get(self._name, None)

    @property
    def __op__(self) -> typing.Callable | NumericType: return self._op

    def __call__(self, *args, **kwargs):
        if isinstance(self._method, str):
            op = getattr(self._op, self._method, None)
        elif self._method is not None:
            raise ValueError(self._method)
        else:
            op = self.__op__

        if not callable(op):
            return op

        try:
            kwargs.update(self._opts)
            value = op(*args, **kwargs)
        except Exception as error:
            raise RuntimeError(
                f"Error when apply  op={op} args={len(args)} kwargs={kwargs}!") from error

        return value


class Derivative(ExprOp):
    def __init__(self, func, order=1, name=None, **kwargs) -> None:
        if name is None:
            name = f"D{list(order)}({func})" if order > 1 else f"D({func})"

        if hasattr(func, "derivative"):
            op = func.derivative(order)
        else:
            op = None
        super().__init__(op, name=name, **kwargs)
        self._func = func
        self._order = order

    def __call__(self, *args, **kwargs):
        if self._op is not None:
            pass
        elif hasattr(self._func, "derivative"):
            self._op = self._func.derivative(self._order)
        return super().__call__(*args, **kwargs)


def derivative(func, *args, **kwargs) -> Derivative:
    return Derivative(func, *args, **kwargs)


class PartialDerivative(ExprOp):
    def __init__(self, func, *order, name=None,  **kwargs) -> None:
        if name is not None:
            pass
        elif len(order) > 0:
            name = name if name is not None else f"d{list(order)}({func})"
        else:
            name = f"d({func})"

        if hasattr(func, "partial_derivative"):
            op = func.partial_derivative(*order)
        else:
            op = None

        super().__init__(op,  name=name, **kwargs)

        self._func = func
        self._order = order

    def __call__(self, *args, **kwargs):
        if self._op is None:
            logger.debug(self._func)
            # op = self._func
            # for i, n in enumerate(self._order):
            #     for j in range(n):
            #         op = jax.grad(op, argument=i)
            # self._op = op

        return super().__call__(*args, **kwargs)


def partial_derivative(func, *args, **kwargs) -> PartialDerivative:
    return PartialDerivative(func, *args, **kwargs)


class Antiderivative(ExprOp):
    def __init__(self,  func, *order, name=None,   **kwargs) -> None:
        if name is not None:
            pass
        elif len(order) > 0:
            name = name if name is not None else f"I{list(order)}({func})"
        else:
            name = f"I({func})"

        if hasattr(func, "antiderivative"):
            op = func.antiderivative(*order)
        else:
            op = None

        super().__init__(op, name=name,  **kwargs)

        self._func = func
        self._order = order

    def __call__(self, *args, **kwargs):
        if self._op is None:
            from ..numlib.interpolate import interpolate
            self._op = interpolate(self._func(*args), *args).antiderivative(*self._order)

        return super().__call__(*args, **kwargs)


def antiderivative(func, *args, **kwargs) -> Antiderivative:
    return Antiderivative(func, *args, **kwargs)


def integral(func, *args, **kwargs) :
    return func. integral(*args, **kwargs)


def find_roots(func, *args, **kwargs) -> typing.Generator[typing.Any, None, None]:
    yield from func.find_roots(*args, **kwargs)
