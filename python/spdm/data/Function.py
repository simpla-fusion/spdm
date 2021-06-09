import collections
import collections.abc
from functools import cached_property
from typing import Any, Callable, Optional, Sequence, Union

from ..util.logger import logger
from ..numlib import interpolate, np, scipy
from .Entry import Entry
from .Node import Node
import warnings


class Function:
    """
        NOTE: Function is imutable!!!!
    """
    def __new__(cls, x, y=None, *args, **kwargs):
        if cls is not Function:
            return object.__new__(cls)

        if isinstance(x, collections.abc.Sequence):
            obj = PiecewiseFunction(x, y, *args, **kwargs)
        # elif not isinstance(x, np.ndarray):
        #     raise TypeError(f"x should be np.ndarray not {type(x)}!")
        else:
            obj = object.__new__(cls)

        return obj

    def __init__(self, x: np.ndarray = None, y: Union[np.ndarray, float, Callable] = None):
        if y is None:
            y = x
            x = None

        if x is not None:
            self._x = np.asarray(x)
        else:
            self._x = None

        if isinstance(y, Node):
            self._y = y._entry.find(default_value=0.0)
        elif isinstance(y, Entry):
            self._y = y.find(default_value=0.0)
        else:
            self._y = y

        # if isinstance(y, Function):
        #     self._y = None
        #     self._func = y
        # elif callable(y):
        #     self._y = None
        #     self._func = y
        # elif isinstance(y, (int, float, np.ndarray, collections.abc.Sequence)):
        #     self._y = np.asarray(y)
        #     self._func = None
        # else:
        #     self._y = None
        #     self._func = func

    @cached_property
    def is_constant(self):
        return isinstance(self._y, (int, float))

    @cached_property
    def is_periodic(self):
        return self.is_constant or (isinstance(self._y, np.ndarray) and np.all(self._y[0] == self._y[-1]))

    @property
    def x(self) -> np.ndarray:
        return self._x

    def duplicate(self):
        return Function(self._x, self._y)

    def __array_ufunc__(self, ufunc, method, *inputs,   **kwargs):
        return Expression(ufunc, method, *inputs, **kwargs)

    def __array__(self) -> np.ndarray:
        if not isinstance(self._y, np.ndarray):
            return np.asarray(self.__call__())
        else:
            return self._y

    @cached_property
    def _ppoly(self) -> interpolate.PPoly:
        d = self.__call__(no_interpolation=True)

        bc_type = getattr(self, '_bc_type', None) or ("periodic" if d[0] == d[-1] else "not-a-knot")

        return interpolate.CubicSpline(self.x, d, bc_type=bc_type)

    def __call__(self, x=None, /, no_interpolation=False, **kwargs):
        if x is None:
            x = self.x

        if isinstance(self._y, (int, float)):
            if x is None or isinstance(x, (float, int)):
                return self._y
            elif isinstance(x, np.ndarray):
                return np.full(x.shape, self._y)
        elif isinstance(self._y, np.ndarray):
            if x is self.x:
                return self._y
            elif not no_interpolation:
                return self._ppoly(x)
            else:
                raise RuntimeError(f"Interpolation is disabled!")
        elif callable(self._y):
            return self._y(x)
        else:
            raise TypeError(f"{type(self._y)}")

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
        elif isinstance(self.x, np.ndarray):
            return self.__call__(self.x[idx])
        else:
            raise RuntimeError(f"x is {type(self.x)} ,y is {type(self._y)}")
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
            return Function(self.x, self._ppoly.derivative()(self.x))
        else:
            return self._ppoly.derivative()(x)

    def antiderivative(self, x=None):
        if x is None:
            return Function(self.x, self._ppoly.antiderivative()(self.x))
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
            return Function(self.__array__(), self.x)
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
    def __init__(self, cond, func, *args,    **kwargs) -> None:
        super().__init__(None, None, *args,    **kwargs)
        self._cond = cond
        self._func = func

    def __array__(self) -> np.ndarray:
        raise NotImplementedError()

    def __len__(self):
        return 0

    def __call__(self, x) -> np.ndarray:
        cond = [c(x) for c in self._cond]
        return np.piecewise(x, cond, self._func)


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
    def x(self) -> np.ndarray:
        try:
            d = next(d.x for d in self._inputs if isinstance(d, Function) and d.x is not None)
        except StopIteration:
            # logger.error(f"{self._ufunc} {[type(d) for d in self._inputs]}")
            # raise RuntimeError(f"Can not get 'x'!")
            d = None
        return d

    def __call__(self, x: Optional[Union[float, np.ndarray]] = None, *args, **kwargs) -> np.ndarray:

        if x is None or (isinstance(x, (collections.abc.Sequence, np.ndarray)) and len(x) == 0):
            x = self.x

        def wrap(x, d):
            if d is None:
                res = 0
            elif isinstance(d, Function):
                res = np.asarray(d(x))
            elif not isinstance(d, np.ndarray) or len(d.shape) == 0:
                res = d
            elif d.shape == self.x.shape:
                res = np.asarray(Function(self.x, d)(x))
            else:
                raise ValueError(f"{self.x.shape} {type(d)} {d.shape}")
            return res
        with warnings.catch_warnings():
            warnings.filterwarnings("always")
            res = self._ufunc(*[wrap(x, d) for d in self._inputs])

        return res

        # if self._method != "__call__":
        #     op = getattr(self._ufunc, self._method)
        #     res = op(*[wrap(x, d) for d in self._inputs])
