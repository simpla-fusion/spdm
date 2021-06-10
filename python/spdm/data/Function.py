import collections
import collections.abc
from functools import cached_property
from operator import is_
from typing import Any, Callable, Optional, Sequence, Type, Union, Set

from numpy.lib.function_base import kaiser


from ..util.logger import logger
from ..numlib import np, scipy
from ..numlib.spline import create_spline, PPoly
from .Entry import Entry
from .Node import Node
import warnings


class Function:
    """
        NOTE: Function is imutable!!!!
    """
    # def __new__(cls, x, y=None, *args, **kwargs):
    #     if cls is not Function:
    #         return object.__new__(cls)

    #     if isinstance(x, collections.abc.Sequence) and isinstance(y, collections.abc.Sequence):
    #         obj = PiecewiseFunction(x, y, *args, **kwargs)
    #     # elif not isinstance(x, np.ndarray):
    #     #     raise TypeError(f"x should be np.ndarray not {type(x)}!")
    #     else:
    #         obj = object.__new__(cls)

    #     return obj
    INIT_NUM_X_POINTS = 64

    def __init__(self, x: Union[np.ndarray, Sequence] = None, y: Union[np.ndarray, float, Callable] = None, /, **kwargs):
        if y is None:
            y = x
            x = None

        if isinstance(y, Node):
            self._y = y._entry.find(default_value=0.0)
        elif isinstance(y, Entry):
            self._y = y.find(default_value=0.0)
        elif isinstance(y, Function):
            if x is None:
                x = y.x_domain
                self._y = y._y
        else:
            self._y = y

        if isinstance(x, np.ndarray):
            self._x_axis = x
            self._x_domain = [x[0], x[-1]]
        elif x is not None:
            self._x_domain = list(set(x))
            self._x_axis = np.linspace(self._x_domain[0], self._x_domain[1], getattr(
                self._y, 'shape', [Function.INIT_NUM_X_POINTS])[0])
        else:
            self._x_domain = [-np.inf, np.inf]
            self._x_axis = None

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
        else:
            if isinstance(self._y,  np.ndarray):
                y = self._y
            elif callable(self._y):
                y = np.asarray(self._y(self.x_axis))
            else:
                raise TypeError(f"{type(self._y)}")

            bc_type = getattr(self, '_bc_type', None) or ("periodic" if np.all(y[0] == y[-1]) else "not-a-knot")
            return create_spline(self.x_axis, y, bc_type=bc_type)

    def __call__(self, x=None, /,  **kwargs) -> Union[np.ndarray, float]:
        if x is not None:
            pass
        elif isinstance(self._y, np.ndarray) and len(kwargs) == 0:
            return self._y
        else:
            x = self.x_axis

        if not isinstance(x, (float, int, np.ndarray)):
            raise TypeError(f"Illegal x {type(x)}!")
        elif isinstance(self._y, (int, float)):
            if isinstance(x, np.ndarray):
                return np.full(x.shape, self._y)
            else:
                return self._y
        elif callable(self._y) and 'dx' not in kwargs:
            return self._y(x, **kwargs)
        else:
            return self._ppoly(x, **kwargs)

    def _resample(self, x_min, x_max, /, tolerance=1.0e-3, init_num_points=16):
        # TODO: Insert points in rapidly changing places.
        x_min = max(self.x_min, x_min)
        x_max = min(self.x_max, x_max)

        if isinstance(self.x_axis, np.ndarray):
            idx_min = np.argmax(self.x_axis >= x_min)
            idx_max = np.argmax(self.x_axis <= x_max)+1
            if isinstance(self._y, np.ndarray):
                return self.x_axis[idx_min:idx_max], self._y[idx_min:idx_max]
            else:
                return self.x_axis[idx_min:idx_max],  self.__call__(self.x_axis[idx_min:idx_max])
        else:
            x_axis = np.linspace(x_min, x_max, Function.INIT_NUM_X_POINTS)
            return x_axis, np.asarray(self.__call__(x_axis))

    def resample(self, x_min, x_max=None, /, **kwargs):
        if x_min is None or (x_max is not None and x_min <= self.x_min and self.x_max <= x_max):
            if len(kwargs) > 0:
                logger.warning(f"ignore key-value arguments {kwargs.keys()}")
            return self
        elif x_max is None:
            return Function(x_min, self.__call__(x_min, **kwargs))
        else:
            return Function(*self._resample(x_min, x_max, **kwargs))

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

    # def __bool__(self) -> bool:
    #     if isinstance(self._y, (int, float)) or ((isinstance(self._y, np.ndarray)) and len(self._y.shape) == 0):
    #         return bool(self._y)
    #     else:
    #         raise TypeError(f"Can not convert {type(self._y)} to float")
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
            v = self._ppoly(self.x)
            x = (self.x[:-1]+self.x[1:])*0.5
            return Function(x, (v[1:]-v[:-1]) / (v[1:]+v[:-1]) / (self.x[1:]-self.x[:-1])*2.0)
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

        return Function(target, self(source).__array__())

    def integrate(self, a=None, b=None):
        return self._ppoly.integrate(a or self.x[0], b or self.x[-1])


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

    @cached_property
    def x_axis(self):
        return NotImplemented

    def __call__(self, x: Union[float, np.ndarray] = None) -> np.ndarray:
        if x is None:
            x = self.x_axis
        if isinstance(x, np.ndarray):
            return np.piecewise(x, [lambda r: self._x[idx] <= r and r < self._x[idx+1] for idx in range(len(self._x)-1)], self._y)
        elif isinstance(x, float):
            idx = next(i for i, val in enumerate(self._x) if val > x)
            return self._y[idx](x)


class Expression(Function):
    def __init__(self, ufunc, method, *inputs,  **kwargs) -> None:

        self._ufunc = ufunc
        self._method = method
        self._inputs = inputs
        self._kwargs = kwargs

        # x = next(d.x for d in self._inputs if isinstance(d, Function))
        # y = None
        super().__init__()

    def __repr__(self) -> str:
        def repr(expr):
            if isinstance(expr, Function):
                return expr.__repr__()
            elif isinstance(expr, np.ndarray):
                return f"<{expr.__class__.__name__} />"
            else:
                return expr

        return f"""<{self.__class__.__name__} op='{self._ufunc.__name__}' > {[repr(a) for a in self._inputs]} </ {self.__class__.__name__}>"""

    @cached_property
    def x_domain(self) -> list:
        res = None
        x_min = None
        x_max = None
        is_changed = False
        for f in self._inputs:
            if not isinstance(f, Function):
                continue
            if res is None:
                res = f.x_domain
                x_min = res[0]
                x_max = res[-1]
            else:
                is_changed = True
                x_min = max(f.x_min, x_min)
                x_max = min(f.x_max, x_max)
                res.extend(f.x_domain)

        if is_changed:
            res = list(set(res))
            res.sort()
            res = [v for v in res if v >= x_min and v <= x_max]
        return res

    @cached_property
    def x_axis(self) -> np.ndarray:
        axis = None
        is_changed = False
        for f in self._inputs:
            if not isinstance(f, Function) or axis is f.x_axis:
                continue
            elif axis is None:
                axis = f.x_axis
            else:
                axis = np.hstack([axis, f.x_axis])
                is_changed = True

        if axis is not None and len(axis) > 0:
            if is_changed:
                axis = np.sort(axis)
                axis = axis[np.append(True, np.diff(axis)) > np.finfo(float).eps*10]
            if axis[0] < self.x_min or axis[-1] > self.x_max:
                axis = np.asarray([v for v in axis if v >= self.x_min and v <= self.x_max])

        return axis

    def __call__(self, x: Optional[Union[float, np.ndarray]] = None, *args, **kwargs) -> np.ndarray:

        if x is None or (isinstance(x, (collections.abc.Sequence, np.ndarray)) and len(x) == 0):
            x = self.x_axis

        def wrap(x, d):
            if d is None:
                res = 0
            elif isinstance(d, Function):
                res = np.asarray(d(x))
            elif not isinstance(d, np.ndarray) or len(d.shape) == 0:
                res = d
            elif self.x_axis is not None and d.shape == self.x_axis.shape:
                res = np.asarray(Function(self.x_axis, d)(x))
            else:
                raise ValueError(f"{getattr(self.x_axis,'shape',[])} {x.shape} {type(d)} {d.shape}")
            return res

        if x is None:
            raise ValueError(type(x))

        with warnings.catch_warnings():
            warnings.filterwarnings("always")
            res = self._ufunc(*[wrap(x, d) for d in self._inputs])

        return res

        # if self._method != "__call__":
        #     op = getattr(self._ufunc, self._method)
        #     res = op(*[wrap(x, d) for d in self._inputs])
