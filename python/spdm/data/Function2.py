import collections
from functools import cached_property

import numpy as np
import scipy.interpolate
from scipy.interpolate.interpolate import PPoly
from typing import Sequence, Union, Optional, Callable, Any

from spdm.data.Function import Expression
from ..util.logger import logger

# from ..data.Quantity import Quantity

# if version.parse(scipy.__version__) <= version.parse("1.4.1"):
#     from scipy.integrate import cumtrapz as cumtrapz
# else:
#     from scipy.integrate import cumulative_trapezoid as cumtrapz

logger.debug(f"SciPy: Version {scipy.__version__}")


class Function:
    def __new__(cls, x, y=None, *args,    **kwargs):
        if cls is not Function:
            return object.__new__(cls)

        if isinstance(x, collections.abc.Sequence):
            obj = PiecewiseFunction(x, y, *args, **kwargs)
        elif not isinstance(x, np.ndarray):
            raise TypeError(f"x should be np.ndarray not {type(x)}!")
        elif y is None and isinstance(x, Function):
            obj = x.duplicate()
        elif isinstance(y, Function):
            obj = y.pullback(x)
        elif isinstance(y, (int, float, complex)):
            obj = ConstantFunc(x, y)
        elif callable(y):
            obj = WrapperFunc(x, y, *args, **kwargs)
        elif isinstance(y, np.ndarray):
            obj = object.__new__(cls)
        else:
            raise RuntimeError(f"{type(x)}   {type(y)} ")

        return obj

    def __init__(self,
                 x: Union[float, np.ndarray, Sequence],
                 y: Optional[Union[float, np.ndarray, Callable, Sequence]] = None,
                 *args, is_periodic=False, **kwargs):
        self._is_periodic = is_periodic
        self._x = x
        self._y = y

    def __array_ufunc__(self, ufunc, method, *inputs,   **kwargs):
        return Expression(ufunc, method, *inputs, **kwargs)

    def __array__(self) -> np.ndarray:
        return self.__call__(self._x)

    def __repr__(self) -> str:
        return self.__array__().__repr__()

    def __getitem__(self, idx, value):
        return self.y[idx]

    def __setitem__(self, idx, value):
        raise NotImplementedError()

    def duplicate(self):
        return Function(self._x, self._y, is_periodic=self._is_periodic)

    @property
    def is_periodic(self):
        return self._is_periodic

    @property
    def x(self):
        return self._x

    @cached_property
    def y(self):
        return self.__array__()

    @cached_property
    def _spl(self):
        return SplineFunction(self.x, self.y, is_periodic=self.is_periodic)

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            args = [self._x]

        if hasattr(self._pimpl, "apply"):
            res = self._pimpl.apply(*args, **kwargs)
        elif callable(self._pimpl):
            res = self._pimpl(*args, **kwargs)
        else:
            res = self.spl.apply(*args, **kwargs)
            # raise RuntimeError(f"{type(self.spl)}")

        return res.view(np.ndarray)

    @cached_property
    def derivative(self):
        return Function(self.spl.derivative)

    @cached_property
    def antiderivative(self):
        return Function(self.spl.antiderivative)

    @cached_property
    def invert(self):
        return Function(self.spl.invert(self._x))

    def pullback(self, *args, **kwargs):
        if len(args) == 0:
            raise ValueError(f"missing arguments!")
        elif len(args) == 2 and args[0].shape == args[1].shape:
            x0, x1 = args
            y = self(x0)
        elif isinstance(args[0], Function) or callable(args[0]):
            logger.warning(f"FIXME: not complete")
            x1 = args[0](self.x)
            y = self.view(np.ndarray)
        elif isinstance(args[0], np.ndarray):
            x1 = args[0]
            y = self(x1)
        else:
            raise TypeError(f"{args}")

        return Function(x1, y, is_periodic=self.is_periodic)

    def integrate(self, a=None, b=None):
        return self.spl.integrate(a or self.x[0], b or self.x[-1])


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
    "__truediv__": np.divide,
    # Return x to the power p, (x**p).
    "__pow__": np.power
}


for name, op in _bi_ops.items():
    setattr(Function,  name, lambda s, other, _op=op: _op(s, other))


class ConstantFunc(Function):
    def __init__(self, x: Any, y: float, *args, is_periodic=True, **kwargs):
        super().__init__(x, y, *args, is_periodic=True, **kwargs)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return np.asarray(self._y)


class WrapperFunc(Function):
    def __init__(self, x, func: Callable, * args,   ** kwargs) -> None:
        super().__init__(x, func, *args, **kwargs)

    def __call__(self, x) -> np.ndarray:
        return self._y(x)


class SplineFunction(Function):
    def __init__(self, x, y=None, is_periodic=False,  ** kwargs) -> None:
        super().__init__()
        self._is_periodic = is_periodic

        if isinstance(x, PPoly) and y is None:
            self._ppoly = x
        elif not isinstance(x, np.ndarray) or y is None:
            raise TypeError((type(x), type(y)))
        else:
            if callable(y):
                y = y(x)
            elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                assert(x.shape == y.shape)
            elif isinstance(y, (float, int)):
                y = np.full(len(x), y)
            else:
                raise NotImplementedError(f"Illegal input {[type(a) for a in args]}")

            if is_periodic:
                self._ppoly = scipy.interpolate.CubicSpline(x, y, bc_type="periodic", **kwargs)
            else:
                self._ppoly = scipy.interpolate.CubicSpline(x, y, **kwargs)

    @property
    def x(self) -> np.ndarray:
        return self._ppoly.x

    @cached_property
    def derivative(self):
        return SplineFunction(self._ppoly.derivative())

    @cached_property
    def antiderivative(self):
        return SplineFunction(self._ppoly.antiderivative())

    def apply(self, x) -> np.ndarray:
        x = self.x if x is None else x
        return self._ppoly(x)

    def integrate(self, a, b):
        return self._ppoly.integrate(a, b)


class PiecewiseFunction(Function):
    def __init__(self, x, cond, func, *args,    **kwargs) -> None:
        super().__init__()
        self._x = x
        self._cond = cond
        self._func = func

    def apply(self, x) -> np.ndarray:
        cond = [c(x) for c in self._cond]
        return np.piecewise(x, cond, self._func)


class Expression(Function):
    def __init__(self, ufunc, method, *inputs,  **kwargs) -> None:

        self._ufunc = ufunc
        self._method = method
        self._inputs = inputs
        self._kwargs = kwargs

        x = next(d.x for d in self._inputs if isinstance(d, Function))
        y = None
        is_periodic = not any([not d.is_periodic for d in self._inputs if isinstance(d, Function)])
        super().__init__(x, y, is_periodic=is_periodic)

    def __call__(self, x: Optional[Union[float, np.ndarray]], *args, **kwargs) -> np.ndarray:
        def wrap(x, d):
            if isinstance(d, Function):
                res = d(x).view(np.ndarray)
            elif not isinstance(d, np.ndarray) or len(d.shape) == 0:
                res = d
            elif d.shape == self.x.shape:
                res = Function(self.x, d)(x).view(np.ndarray)
            else:
                raise ValueError(f"{self.x.shape} {d.shape}")

            return res

        if self._method != "__call__":
            op = getattr(self._ufunc, self._method)
            # raise RuntimeError((self._ufunc, self._method))
            res = op(*[wrap(x, d) for d in self._inputs])
        try:
            res = self._ufunc(*[wrap(x, d) for d in self._inputs])
        except Warning as error:
            raise ValueError(
                f"\n {self._ufunc}  {[type(a) for a in self._inputs]}  {[a.shape for a in self._inputs if isinstance(a,Function)]} {error} \n ")
        return res
