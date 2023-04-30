from __future__ import annotations

import collections.abc
from functools import cached_property
import typing
import numpy as np

from ..utils.logger import logger
from ..utils.misc import float_unique
from ..utils.tags import _not_found_
from ..grid.Grid import Grid


_T = typing.TypeVar("_T")


class Function(typing.Generic[_T]):
    """
        NOTE: Function is immutable!!!!
    """

    def __init__(self, data, *args, **kwargs):

        if callable(data):
            self.__fun_op__ = data
            self._data = None
        else:
            self.__fun_op__ = None
            self._data = data

        if len(args) == 1 and isinstance(args[0], np.ndarray):
            kwargs.setdefault("grid_type", "Spline1D")

        self._grid = Grid(*args, **kwargs)

    @property
    def grid(self) -> Grid | None:
        return self._grid

    def __get_data__(self) -> _T | np.ndarray | None:
        if hasattr(self._data.__class__, "__entry__"):
            d = self._data.__entry__().__value__()
            if d is not None and d is not _not_found_:
                self._data = d
        else:
            d = self._data

        return d

    def __array__(self) -> np.ndarray:
        return np.asarray(self.__get_data__())

    def __array_ufunc__(self, ufunc, method, *inputs,   **kwargs):
        return Expression(ufunc, method, *inputs, **kwargs)

    def __call__(self, *args) -> _T | np.ndarray:
        if callable(self.__fun_op__):
            return self.__fun_op__(*args)

        if self._grid is not None:
            self.__fun_op__ = self._grid.interpolator(self.__get_data__())
        elif isinstance(self._data, (int, float, bool, complex)):
            self.__fun_op__ = lambda *_: self.__type_hint__(self.__get_data__())
        else:
            raise RuntimeError(f"Function is not callable! {self.__fun_op__}")

        return self.__fun_op__(*args)

    def __type_hint__(self) -> typing.Type:
        return typing.get_args(self.__orig_class__)[0]

    def __duplicate__(self):
        other = object.__new__(self.__class__)
        other._grid = self._grid
        other._data = self._data.copy() if isinstance(self._data, np.ndarray) else self._data
        return other

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}  grid_type={self._grid.type if self._grid is not None else 'None'} data_type={self.__type_hint__.__name__}/>"

    def __getitem__(self, *args):
        value = self.__value__()
        if isinstance(value, np.ndarray):
            return value.__getitem__(*args)
        elif isinstance(self._grid, np.ndarray):
            return self.__call__(*self._grid.__getitem__(*args))
        else:
            raise RuntimeError(f"x is {type(self._grid)} ,y is {type(self._data)}")

    def derivative(self) -> Function:
        return Function(self.__fun_op__.derivative())

    def antiderivative(self) -> Function:
        return Function(self.__fun_op__.antiderivative())

    def dln(self) -> Function:
        # v = self._interpolator(self._grid)
        # x = (self._grid[:-1]+self._grid[1:])*0.5
        # return Function(x, (v[1:]-v[:-1]) / (v[1:]+v[:-1]) / (self._grid[1:]-self._grid[:-1])*2.0)
        return Function(self._grid, self.__fun_op__.derivative()(self._grid)/self.__fun_op__(self._grid))

    def integrate(self, a=None, b=None):
        return self.__fun_op__.integrate(a or self.x[0], b or self.x[-1])

    def invert(self) -> Function:
        return Function(self.__array__(), self._grid)

    def pullback(self, source: np.ndarray, target: np.ndarray):
        if source.shape != target.shape:
            raise ValueError(
                f"The shapes of axies don't match! {source.shape}!={target.shape}")

        return Function(target, np.asarray(self(source), dtype=float))


def function_like(*args, **kwargs) -> Function:
    if len(args) == 1 and isinstance(args[0], Function):
        return args[0]
    else:
        return Function(*args, **kwargs)


# __op_list__ = ['abs', 'add', 'and',
#                #  'attrgetter',
#                'concat',
#                # 'contains', 'countOf',
#                'delitem', 'eq', 'floordiv', 'ge',
#                # 'getitem',
#                'gt',
#                'iadd', 'iand', 'iconcat', 'ifloordiv', 'ilshift', 'imatmul', 'imod', 'imul',
#                'index', 'indexOf', 'inv', 'invert', 'ior', 'ipow', 'irshift',
#                #    'is_', 'is_not',
#                'isub',
#                # 'itemgetter',
#                'itruediv', 'ixor', 'le',
#                'length_hint', 'lshift', 'lt', 'matmul',
#                #    'methodcaller',
#                'mod',
#                'mul', 'ne', 'neg', 'not', 'or', 'pos', 'pow', 'rshift',
#                #    'setitem',
#                'sub', 'truediv', 'truth', 'xor']
_uni_ops = {
    '__neg__': np.negative,
}

for names, op in _uni_ops.items():
    setattr(Function,  names, lambda s, _op=op: _op(s))

_bi_ops = {

    # Add arguments element-wise.
    "__add__": np.add,
    # (x1, x2, / [, out, where, casting, …]) Subtract arguments, element-wise.
    "__sub__": np.subtract,
    # multiply(x1, x2, / [, out, where, casting, …])  Multiply arguments element-wise.
    "__mul__": np.multiply,
    # (x1, x2, / [, out, casting, order, …])   Matrix product of two arrays.
    "__matmul__": np.matmul,
    # (x1, x2, / [, out, where, casting, …])   Returns a true division of the inputs, element-wise.
    "__truediv__": np.true_divide,
    # Return x to the power p, (x**p).
    "__pow__": np.power,
    "__eq__": np.equal,
    "__ne__": np.not_equal,
    "__lt__": np.less,
    "__le__": np.less_equal,
    "__gt__": np.greater_equal,
    "__ge__": np.greater_equal,
}

for names, op in _bi_ops.items():
    setattr(Function,  names, lambda s, other, _op=op: _op(s, other))

_rbi_ops = {
    # Add arguments element-wise.
    "__radd__": np.add,
    # (x1, x2, / [, out, where, casting, …]) Subtract arguments, element-wise.
    "__rsub__": np.subtract,
    # multiply(x1, x2, / [, out, where, casting, …])  Multiply arguments element-wise.
    "__rmul__": np.multiply,
    # (x1, x2, / [, out, casting, order, …])   Matrix product of two arrays.
    "__rmatmul__": np.matmul,
    # (x1, x2, / [, out, where, casting, …])   Returns a true division of the inputs, element-wise.
    "__rtruediv__": np.divide,
    # Return x to the power p, (x**p).
    "__rpow__": np.power
}

for names, op in _rbi_ops.items():
    setattr(Function,  names, lambda s, other, _op=op: _op(other, s))


class Expression(Function):
    def __init__(self, ufunc, method, *inputs,  **kwargs) -> None:
        super().__init__(*inputs)
        self._ufunc = ufunc
        self._method = method
        self._kwargs = kwargs

    def __repr__(self) -> str:
        def repr(expr):
            if isinstance(expr, Function):
                return expr.__repr__()
            elif isinstance(expr, np.ndarray):
                return f"<{expr.__class__.__name__} />"
            else:
                return expr

        return f"""<{self.__class__.__name__} op='{self._ufunc.__name__}' > {[repr(a) for a in self._data]} </ {self.__class__.__name__}>"""

    @cached_property
    def x_domain(self) -> list:
        res = []
        x_min = -np.inf
        x_max = np.inf
        for f in self._data:
            if not isinstance(f, Function) or f.x_domain is None:
                continue
            x_min = max(f.x_min, x_min)
            x_max = min(f.x_max, x_max)
            res.extend(f.x_domain)
        return float_unique(res, x_min, x_max)

    @cached_property
    def x_axis(self) -> np.ndarray:
        axis = None
        is_changed = False
        for f in self._data:
            if not isinstance(f, Function) or f.x_axis is None or axis is f.x_axis:
                continue
            elif axis is None:
                axis = f.x_axis
            else:
                axis = np.hstack([axis, f.x_axis])
                is_changed = True

        if is_changed:
            axis = float_unique(axis, self.x_min, self.x_max)
        return axis

    def resample(self, x_min, x_max=None, /, **kwargs):
        inputs = [(f.resample(x_min, x_max, **kwargs)
                   if isinstance(f, Function) else f) for f in self._data]
        return Expression(self._ufunc, self._method, *inputs, **self._kwargs)

    def __call__(self, x: typing.Optional[typing.Union[float, np.ndarray]] = None, *args, **kwargs) -> np.ndarray:

        if x is None or (isinstance(x, list) and len(x) == 0):
            x = self._grid

        if x is None:
            raise RuntimeError(f"Can not get x_axis!")

        def wrap(x, d):
            if d is None:
                res = 0
            elif isinstance(d, Function):
                res = np.asarray(d(x), dtype=float)
            elif not isinstance(d, np.ndarray) or len(d.shape) == 0:
                res = d
            elif self._grid is not None and d.shape == self._grid.shape:
                res = np.asarray(Function(self._grid, d)(x), dtype=float)
            elif d.shape == x.shape:
                res = d
            else:
                raise ValueError(
                    f"{getattr(self._grid,'shape',[])} {x.shape} {type(d)} {d.shape}")
            return res

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                res = self._ufunc(*[wrap(x, d) for d in self._data])
            except RuntimeWarning as warning:
                logger.error((self._ufunc, [wrap(x, d) for d in self._data]))
                logger.exception(warning)
                raise RuntimeError(warning)
        return res

        # if self._method != "__call__":
        #     op = getattr(self._ufunc, self._method)
        #     res = op(*[wrap(x, d) for d in self._data])
