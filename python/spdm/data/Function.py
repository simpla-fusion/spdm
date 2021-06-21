import collections
import collections.abc
import warnings
from functools import cached_property
from logging import log
from operator import is_
from typing import Any, Callable, Optional, Sequence, Set, Type, Union

from ..numlib import np, scipy
from ..numlib.misc import float_unique
from ..numlib.spline import PPoly, create_spline
from ..util.logger import logger
from .Entry import Entry
from .Node import Node


class Function:
    """
        NOTE: Function is imutable!!!!
    """

    def __init__(self, x: Union[np.ndarray, Sequence] = None, y: Union[np.ndarray, float, Callable] = None, /, **kwargs):
        if y is None:
            y = x
            x = None

        if y is None or (isinstance(y, (np.ndarray, collections.abc.Sequence)) and len(y) == 0):
            raise ValueError(f"y is None!")
        elif isinstance(y, Node):
            self._y = y._entry.find(default_value=0.0)
        elif isinstance(y, Entry):
            self._y = y.find(default_value=0.0)
        elif isinstance(y, Function):
            if x is None:
                x = y.x_domain
                self._y = y._y
        elif isinstance(y, PPoly):
            self._y = y
            if x is None:
                x = y.x
        else:
            self._y = y

        if x is None or len(x) == 0:
            self._x_axis = None
            self._x_domain = [-np.inf, np.inf]
        elif isinstance(x, np.ndarray):
            if len(x) == 0:
                logger.error(f"{type(x)} {type(y)}")
            self._x_axis = x
            self._x_domain = [x[0], x[-1]]
        elif isinstance(x, collections.abc.Sequence) and len(x) > 0:
            self._x_domain = list(set(x))
            self._x_axis = None
        else:
            self._x_domain = [-np.inf, np.inf]
            self._x_axis = None

        if isinstance(self._y, np.ndarray) and self._x_axis is None:
            raise ValueError(f"x_axis is None")

    @property
    def is_valid(self) -> bool:
        return self._x_axis is not None and self._y is not None

    @cached_property
    def is_constant(self) -> bool:
        return isinstance(self._y, (int, float))

    @cached_property
    def is_periodic(self) -> bool:
        return self.is_constant \
            or (isinstance(self._y, np.ndarray) and np.all(self._y[0] == self._y[-1])) \
            or np.all(self.__call__(self.x_min) == self.__call__(self.x_max))

    @property
    def is_bounded(self):
        return not (self.x_min == -np.inf or self.x_max == np.inf)

    @property
    def continuous(self) -> bool:
        return len(self.x_domain) == 2

    @property
    def x_domain(self) -> list:
        return self._x_domain

    @property
    def x_min(self) -> float:
        return self.x_domain[0]

    @property
    def x_max(self) -> float:
        return self.x_domain[-1]

    @property
    def x_axis(self) -> np.ndarray:
        return self._x_axis

    def __len__(self) -> int:
        if self.x_axis is not None:
            return len(self.x_axis)
        else:
            raise RuntimeError(f"Can not get length from {type(self._y)} or {type(self.x_axis)}")

    def duplicate(self):
        return self.__class__(self.x_axis, self._y)

    def __array_ufunc__(self, ufunc, method, *inputs,   **kwargs):
        return Expression(ufunc, method, *inputs, **kwargs)

    def __array__(self) -> np.ndarray:
        return self._y if isinstance(self._y, np.ndarray) else np.asarray(self.__call__())

    @cached_property
    def _ppoly(self) -> PPoly:

        if isinstance(self._y,  PPoly):
            return self._y
        elif self.x_axis is None:
            raise ValueError(f"x_axis is None")
        elif isinstance(self._y, np.ndarray):
            return create_spline(self.x_axis,  self._y)
        else:
            return create_spline(self.x_axis,  self.__call__())

    def __call__(self, x=None, /,  **kwargs) -> Union[np.ndarray, float]:
        if x is None:
            x = self._x_axis

        if x is None:
            raise RuntimeError(f"x_axis is None!")
        _y = getattr(self, "_y", None)
        if x is self._x_axis and isinstance(_y, np.ndarray):
            return self._y

        if _y is None:
            raise RuntimeError(f"Illegal function! y is None {self.__class__}")
        elif isinstance(_y, (int, float)):
            if isinstance(x, np.ndarray):
                return np.full(x.shape, _y)
            else:
                return _y
        elif callable(_y):
            return np.asarray(_y(x, **kwargs))
        elif x is not self._x_axis and isinstance(_y, np.ndarray):
            return self._ppoly(x, **kwargs)
        else:
            raise TypeError((type(x), type(_y)))

    def resample(self, x_min, x_max=None, /, **kwargs):
        if x_min is None or (x_max is not None and x_min <= self.x_min and self.x_max <= x_max):
            if len(kwargs) > 0:
                logger.warning(f"ignore key-value arguments {kwargs.keys()}")
                # TODO: Insert points in rapidly changing places.
            return self
        elif x_max is None:
            return Function(x_min, self.__call__(x_min, **kwargs))

        x_min = max(self.x_min, x_min)
        x_max = min(self.x_max, x_max)

        if x_min > x_max or np.isclose(x_min, x_max) or x_max <= self.x_min:
            raise ValueError(f"{x_min,x_max}  not in  {self.x_min,self.x_max}")
        elif isinstance(self.x_axis, np.ndarray):
            idx_min = np.argmax(self.x_axis >= x_min)
            idx_max = np.argmax(self.x_axis > x_max)
            if idx_max > idx_min:
                pass
            elif idx_max == 0 and np.isclose(self.x_axis[-1], x_max):
                idx_max = -1
            else:
                logger.debug((x_min, x_max, idx_min, idx_max, self.x_axis))
            if isinstance(self._y, np.ndarray):
                return Function(self.x_axis[idx_min:idx_max], self._y[idx_min:idx_max])
            else:
                return Function(self.x_axis[idx_min:idx_max],  self.__call__(self.x_axis[idx_min:idx_max]))
        elif callable(self._y):
            return Function([x_min, x_max], self._y)
        else:
            raise TypeError((type(self.x_axis), type(self._y)))
            # return x_axis, np.asarray(self.__call__(x_axis))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}  type={type(self._y)}/>"

    def __float__(self) -> float:
        if isinstance(self._y, (int, float)) or ((isinstance(self._y, np.ndarray)) and len(self._y.shape) == 0):
            return float(self._y)
        else:
            raise TypeError(f"Can not convert {type(self._y)} to float. {getattr(self._y,'shape',None)}")

    def __int__(self) -> int:
        if isinstance(self._y, (int, float)) or ((isinstance(self._y, np.ndarray)) and len(self._y.shape) == 0):
            return int(self._y)
        else:
            raise TypeError(f"Can not convert {type(self._y)} to float")

    def __getitem__(self, idx):
        if isinstance(self._y, np.ndarray):
            return self._y[idx]
        elif isinstance(self.x_axis, np.ndarray):
            return self.__call__(self.x_axis[idx])
        else:
            raise RuntimeError(f"x is {type(self.x_axis)} ,y is {type(self._y)}")

    # def __setitem__(self, idx, value):
    #     if hasattr(self, "_ppoly"):
    #         delattr(self, "_ppoly")
    #     self.__real_array__()[idx] = value
    # def _prepare(self, x, y):
    #     if not isinstance(x, [collections.abc.Sequence, np.ndarray]):
    #         x = np.asarray([x])
    #     if isinstance(y, [collections.abc.Sequence, np.ndarray]):
    #         y = np.asarray(y)
    #     elif y is not None:
    #         y = np.asarray([y])
    #     else:
    #         y = self.__call__(x)
    #     return x, y
    # def insert(self, x, y=None):
    #     res = Function(*self._prepare(x, y), func=self._func)
    #     raise NotImplementedError('Insert points!')
    #     return res
    # def __len__(self):
    #     return len(self.x) if self.x is not None else 0

    def derivative(self, x=None):
        if x is None:
            return Function(self._ppoly.derivative())
        else:
            return self._ppoly.derivative()(x)

    def antiderivative(self, x=None):
        if x is None:
            return Function(self._ppoly.antiderivative())
        else:
            return self._ppoly.antiderivative()(x)

    def dln(self, x=None):
        if x is None:
            v = self._ppoly(self.x_axis)
            x = (self.x_axis[:-1]+self.x_axis[1:])*0.5
            return Function(x, (v[1:]-v[:-1]) / (v[1:]+v[:-1]) / (self.x_axis[1:]-self.x_axis[:-1])*2.0)
            # return Function(self.x, self._ppoly.derivative()(self.x)/self._ppoly(self.x))
        else:
            return self.dln()(x)
            # v = self._ppoly(x)
            # return Function((x[:-1]+x[1:])*0.5, (v[1:]-v[:-1]) / (v[1:]+v[:-1]) / (x[1:] - x[:-1])*2.0)
            # return self._ppoly.derivative()(x)/self._ppoly(x)

    def invert(self, x=None):
        if x is None:
            return Function(self.__array__(), self.x_axis)
        else:
            return Function(self.__call__(x), x)

    def pullback(self, source: np.ndarray, target: np.ndarray):
        if source.shape != target.shape:
            raise ValueError(f"The shapes of axies don't match! {source.shape}!={target.shape}")

            # if len(args) == 0:
            #     raise ValueError(f"missing arguments!")
            # elif len(args) == 2 and args[0].shape == args[1].shape:
            #     x0, x1 = args
            #     y = self(x0)
            # elif isinstance(args[0], Function) or callable(args[0]):
            #     logger.warning(f"FIXME: not complete")
            #     x1 = args[0](self.x)
            #     y = self.view(np.ndarray)
            # elif isinstance(args[0], np.ndarray):
            #     x1 = args[0]
            #     y = self(x1)
            # else:
            #     raise TypeError(f"{args}")

        return Function(target, np.asarray(self(source)))

    def integrate(self, a=None, b=None):
        return self._ppoly.integrate(a or self.x[0], b or self.x[-1])


def function_like(x, y) -> Function:
    if isinstance(y, Function):
        return y
    else:
        return Function(x, y)
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

for name, op in _uni_ops.items():
    setattr(Function,  name, lambda s, _op=op: _op(s))

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

for name, op in _bi_ops.items():
    setattr(Function,  name, lambda s, other, _op=op: _op(s, other))

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

for name, op in _rbi_ops.items():
    setattr(Function,  name, lambda s, other, _op=op: _op(other, s))


class PiecewiseFunction(Function):
    def __init__(self, x, y, *args,    **kwargs) -> None:
        super().__init__(x, y, *args,    **kwargs)
        assert(len(x) == len(y)+1)

    def resample(self, x_min, x_max=None, /, **kwargs):
        x_min = x_min or -np.inf
        x_max = x_max or np.inf
        if x_min <= self.x_min and x_max >= self.x_max:
            return self
        cond_list = []
        func_list = []
        for idx, xp in enumerate(self.x_domain[:-1]):
            if x_max <= xp:
                break
            elif x_min >= self.x_domain[idx+1]:
                continue

            if x_min <= xp:
                cond_list.append(xp)
                func_list.append(self._y[idx])
            if x_max < self.x_domain[idx+1]:
                break
        if len(cond_list) == 0:
            return None
        else:
            cond_list.append(min(x_max, self.x_domain[-1]))
            return PiecewiseFunction(cond_list, func_list)

    def __call__(self, x: Union[float, np.ndarray] = None) -> np.ndarray:
        if x is None:
            x = self.x_axis
        elif not isinstance(x, (int, float, np.ndarray)):
            x = np.asarray(x)

        if isinstance(x, np.ndarray) and len(x) == 1:
            x = x[0]

        if isinstance(x, np.ndarray):
            cond_list = [np.logical_and(self.x_domain[idx] <= x, x < self.x_domain[idx+1])
                         for idx in range(len(self.x_domain)-1)]
            cond_list[-1] = np.logical_or(cond_list[-1], np.isclose(x, self.x_domain[-1]))
            return np.piecewise(x, cond_list, self._y)
        elif isinstance(x, (int, float)):

            if np.isclose(x, self.x_domain[0]):
                idx = 0
            elif np.isclose(x, self.x_domain[-1]):
                idx = -1
            else:
                try:
                    idx = next(i for i, val in enumerate(self.x_domain) if val >= x)-1
                except StopIteration:
                    idx = None
            if idx is None:
                raise ValueError(f"Out of range! {x} not in ({self.x_domain[0]},{self.x_domain[-1]})")

            return self._y[idx](x)
        else:
            raise TypeError(type(x))


class Expression(Function):
    def __init__(self, ufunc, method, *inputs,  **kwargs) -> None:
        super().__init__(inputs)
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

        return f"""<{self.__class__.__name__} op='{self._ufunc.__name__}' > {[repr(a) for a in self._y]} </ {self.__class__.__name__}>"""

    @cached_property
    def x_domain(self) -> list:
        res = []
        x_min = -np.inf
        x_max = np.inf
        for f in self._y:
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
        for f in self._y:
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
        inputs = [(f.resample(x_min, x_max, **kwargs) if isinstance(f, Function) else f) for f in self._y]
        return Expression(self._ufunc, self._method, *inputs, **self._kwargs)

    def __call__(self, x: Optional[Union[float, np.ndarray]] = None, *args, **kwargs) -> np.ndarray:

        if x is None or (isinstance(x, (collections.abc.Sequence, np.ndarray)) and len(x) == 0):
            x = self.x_axis

        if x is None:
            raise RuntimeError(f"Can not get x_axis!")

        def wrap(x, d):
            if d is None:
                res = 0
            elif isinstance(d, Function):
                res = np.asarray(d(x))
            elif not isinstance(d, np.ndarray) or len(d.shape) == 0:
                res = d
            elif self.x_axis is not None and d.shape == self.x_axis.shape:
                res = np.asarray(Function(self.x_axis, d)(x))
            elif d.shape == x.shape:
                res = d
            else:
                raise ValueError(f"{getattr(self.x_axis,'shape',[])} {x.shape} {type(d)} {d.shape}")
            return res

        with warnings.catch_warnings():
            warnings.filterwarnings("always")
            res = self._ufunc(*[wrap(x, d) for d in self._y])

        return res

        # if self._method != "__call__":
        #     op = getattr(self._ufunc, self._method)
        #     res = op(*[wrap(x, d) for d in self._y])
