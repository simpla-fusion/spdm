from __future__ import annotations

import typing
import numpy as np
from ..utils.logger import logger
from ..utils.typing import (NumericType, numeric_type)
ExprOpLike = typing.Callable | None


class ExprOp:
    """
        算符: 用于表示一个运算符，可以是函数，也可以是类的成员函数
        受 np.ufunc 启发而来。
        可以通过 ExprOp(op, method=method) 的方式构建一个 ExprOp 对象。
    """

    def __init__(self, op: typing.Callable | None, /,
                 method: str | None = None,
                 **kwargs) -> None:
        self._op = op
        self._method = method
        self._opts = kwargs

    @property
    def __label__(self) -> str:
        if isinstance(self._op, np.ufunc):
            return self._op.__name__
        else:
            return str(self._op)

    def __str__(self) -> str:
        if isinstance(self._op, np.ufunc):
            return self._op.__name__
        else:
            return str(self._op)

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

    def derivative(self, *args, **kwargs): return Derivative(self, *args, **kwargs)

    def partial_derivative(self, *args, **kwargs): return PartialDerivative(self, *args, **kwargs)

    def antiderivative(self, *args, **kwargs): return Antiderivative(self, *args, **kwargs)

    def interpolate(self, *args, **kwargs):
        from ..numlib.interpolate import interpolate
        return interpolate(self, *args, **kwargs)


def as_exprop(expr, *args, **kwargs) -> ExprOp | None:
    if isinstance(expr, ExprOp):
        return expr
    elif callable(expr):
        return ExprOp(expr, *args, **kwargs)
    elif expr is None:
        return None
    else:
        raise TypeError(f"expr={expr} is not callable!")


class Derivative(ExprOp):
    def __init__(self, order, expr: ExprOp, *args,    **kwargs) -> None:
        super().__init__(None,  *args, **kwargs)
        self._expr = expr
        self._order = order

    def __str__(self) -> str:
        if len(self._order) > 0:
            return f"D{list(self._order)}({self._expr})"
        else:
            return f"D({self._expr})"

    def __call__(self, *args, **kwargs): return super().__call__(*args, **kwargs)


def derivative(order, expr: ExprOp, *args, **kwargs) -> Derivative:
    if not isinstance(expr, ExprOp):
        return 0.0
    else:
        return Derivative(order, expr, *args, **kwargs)


class PartialDerivative(ExprOp):
    def __init__(self, order, expr: ExprOp, *args, **kwargs) -> None:
        super().__init__(None, **kwargs)

        self._expr = expr
        self._order = order

    def __str__(self) -> str:
        if len(self._order) > 0:
            return f"d{list(self._order)}({self._expr})"
        else:
            return f"d({self._expr})"

    def __call__(self, *args, **kwargs):
        # if self._op is None:
        # op = self._func
        # for i, n in enumerate(self._order):
        #     for j in range(n):
        #         op = jax.grad(op, argument=i)
        # self._op = op

        return super().__call__(*args, **kwargs)


def partial_derivative(order, expr: ExprOp, *args, **kwargs) -> PartialDerivative:
    if not isinstance(expr, ExprOp):
        return 0.0
    else:
        return PartialDerivative(order, expr, *args, **kwargs)


class Antiderivative(ExprOp):
    def __init__(self,  order, expr: ExprOp,    **kwargs) -> None:

        super().__init__(None,   **kwargs)

        self._expr = expr
        self._order = order

    def _repr_latex_(self) -> str:
        if len(self._order) > 0:
            return f"I{list(self._order)}({self._expr})"
        else:
            return f"I({self._expr})"

    def __call__(self, *args, **kwargs):
        if self._op is None:
            from ..numlib.interpolate import interpolate
            self._op = interpolate(self._expr(*args), *args).antiderivative(*self._order)

        return super().__call__(*args, **kwargs)


def antiderivative(order, func, *args, **kwargs) -> Antiderivative:
    if not isinstance(func, ExprOp):
        raise TypeError(f"func={func} is not a ExprOp!")

    return Antiderivative(order, func, *args, **kwargs)


def integral(func, *args, **kwargs):
    return func.integral(*args, **kwargs)


def find_roots(func, *args, **kwargs) -> typing.Generator[typing.Any, None, None]:
    yield from func.find_roots(*args, **kwargs)
