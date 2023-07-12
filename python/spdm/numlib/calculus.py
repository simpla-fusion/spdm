from ..data.Functor import Functor
from ..data.Expression import Expression
from ..utils.typing import NumericType
import typing
from .interpolate import interpolate


class Derivative(Functor):
    def __init__(self,  expr: Functor | Expression, *args,    **kwargs) -> None:
        super().__init__(None,   **kwargs)
        self._expr = expr
        self._order = args

    def __str__(self) -> str:
        if len(self._order) > 0:
            return f"D{list(self._order)}({self._expr})"
        else:
            return f"D({self._expr})"

    def __call__(self, *args, **kwargs):
        if self._op is None:
            value = self._expr(*args)
            self._op = interpolate(value, *args, **kwargs).derivative(*self._order)
        return super().__call__(*args, **kwargs)


def derivative(expr: Functor | Expression,  *args, **kwargs) -> Expression:
    return Expression(Derivative(expr,  *args), ** kwargs)


class PartialDerivative(Functor):
    def __init__(self, expr: Functor | Expression, *args, **kwargs) -> None:
        super().__init__(None, **kwargs)
        self._expr = expr
        self._order = args

    def __call__(self, *args, **kwargs):
        if self._op is None:
            value = self._expr(*args)
            self._op = interpolate(value, *args, **kwargs).partial_derivative(*self._order)
        return super().__call__(*args, **kwargs)


def partial_derivative(expr: Functor | Expression, *args, **kwargs) -> Expression:
    return Expression(PartialDerivative(expr, *args), **kwargs)


class Antiderivative(Functor):
    def __init__(self,  expr: Functor | Expression,   *args,    **kwargs) -> None:

        super().__init__(None,   **kwargs)

        self._expr = expr
        self._order = args

    def _repr_latex_(self) -> str:
        if len(self._order) > 0:
            return f"I{list(self._order)}({self._expr})"
        else:
            return f"I({self._expr})"

    def __call__(self, *args, **kwargs):
        if self._op is None:
            value = self._expr(*args)
            self._op = interpolate(value, *args, **kwargs).antiderivative(*self._order)
        return super().__call__(*args, **kwargs)


def antiderivative(expr: Expression, *args, **kwargs) -> Expression:
    return Expression(Antiderivative(expr, *args), **kwargs)


def integral(func, *args, **kwargs):
    return func.integral(*args, **kwargs)


def find_roots(func, *args, **kwargs) -> typing.Generator[typing.Any, None, None]:
    yield from func.find_roots(*args, **kwargs)
