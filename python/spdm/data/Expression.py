from __future__ import annotations

import typing
from copy import deepcopy, copy

import numpy as np
import numpy.typing as np_tp

from ..utils.logger import logger
from ..utils.numeric import float_nan
from ..utils.tags import _not_found_
from ..utils.typing import (ArrayType, NumericType, array_type, as_array,
                            is_scalar, numeric_type)
from ..views.View import display
from .Functor import Functor


class Expression:
    """
        Expression
        -----------
        表达式是由多个操作数和运算符按照约定的规则构成的一个序列。
        其中运算符表示对操作数进行何种操作，而操作数可以是变量、常量、数组或者表达式。
        表达式可以理解为树状结构，每个节点都是一个操作数或运算符，每个节点都可以有多个子节点。
        表达式的值可以通过对树状结构进行遍历计算得到。
        没有子节点的节点称为叶子节点，叶子节点可以是常量、数组，也可以是变量和函数。

        变量是一种特殊的函数，它的值由上下文决定。

        例如：
            >>> import spdm
            >>> x = spdm.data.Expression(op=np.sin)
            >>> y = spdm.data.Expression(op=np.cos)
            >>> z = x + y
            >>> z
            <Expression   op="add" />
            >>> z(0.0)
            3.0
    """

    fill_value = float_nan

    def __init__(self, expr: Functor | Expression | None = None, *children, label: str = None, **kwargs) -> None:
        """
            Parameters
            ----------
            args : typing.Any
                操作数
            op : typing.Callable  | ExprOp
                运算符，可以是函数，也可以是类的成员函数
            kwargs : typing.Any
                命名参数， 用于传递给运算符的参数

        """

        if isinstance(expr, Expression) and len(children) == 0:
            self.__copy_from__(expr)

        elif expr is None or isinstance(expr, Functor):
            pass

        elif callable(expr):
            expr = Functor(expr)

        elif not isinstance(expr,  Functor):
            raise NotImplementedError(f"{type(expr)}")

        self._func = expr
        self._children = children
        self._label = label

        if len(kwargs) > 0:
            logger.warning(f"Expression.__init__() ignore kwargs {kwargs}")

    def __copy__(self) -> Expression:
        """ 复制一个新的 Expression 对象 """
        other: Expression = object.__new__(self.__class__)
        other.__copy_from__(self)
        return other

    def __copy_from__(self, other: Expression) -> Expression:
        """ 复制 other 到 self  """
        if isinstance(other, Expression):
            self._func = copy(other._func)
            self._children = copy(other._children)
            self._label = other._label
        return self

    def __array_ufunc__(self, ufunc, method, *args, **kwargs) -> Expression:
        """
            重载 numpy 的 ufunc 运算符, 用于在表达式中使用 numpy 的 ufunc 函数构建新的表达式。
            例如：
                >>> import numpy as np
                >>> import spdm
                >>> x = spdm.data.Expression(np.sin)
                >>> y = spdm.data.Expression(np.cos)
                >>> z = x + y
                >>> z
                <Expression   op="add" />
                >>> z(0.0)
                1.0
        """
        return Expression(Functor(ufunc, method=method, **kwargs), *args)

    def __array__(self) -> ArrayType: return as_array(self.__call__())

    @property
    def has_children(self) -> bool: return len(self._children) > 0
    """ 判断是否有子节点"""

    @property
    def empty(self) -> bool: return not self.has_children and self._func is None

    @property
    def callable(self): return callable(self._func) or self.has_children

    @property
    def __label__(self) -> str: return self._label

    def __str__(self): return self.__label__ or self.__class__.__name__

    def _repr_latex_(self): return display(self, backend="latex")
    """ for jupyter notebook display """

    @property
    def dtype(self): return self._type_hint()

    def _type_hint(self, *args): return float
    """ TODO:获取表达式的类型 """

    def __domain__(self, *x) -> bool | np_tp.NDArray[np.bool_]:
        """ 当坐标在定义域内时返回 True，否则返回 False  """

        d = [child.__domain__(*x) for child in self._children if hasattr(child, "__domain__")]

        if isinstance(self._func, Functor):
            d += [self._func.__domain__(*x)]

        d = [v for v in d if (v is not None and v is not True)]

        if len(d) > 0:
            return np.bitwise_and.reduce(d)
        else:
            return True

    def __functor__(self) -> Functor: return self._func
    """ 获取表达式的运算符，若为 constants 函数则返回函数值 """

    def __call__(self, *xargs: NumericType, **kwargs) -> typing.Any:
        """
            重载函数调用运算符，用于计算表达式的值

            TODO:
            - support JIT compilation
            - support broadcasting?
            - support multiple meshes?

            Parameters
            ----------
            xargs : NumericType
                自变量/坐标，可以是标量，也可以是数组
            kwargs : typing.Any
                命名参数，用于传递给运算符的参数
        """

        if len(xargs) == 0:
            return self
        elif any([(isinstance(arg, Expression) or callable(arg)) for arg in xargs]):
            return Expression(Functor(self, **kwargs), *xargs)

        # 根据 __domain__ 函数的返回值，对输入坐标进行筛选

        mark = self.__domain__(*xargs)

        mark_size = mark.size if isinstance(mark, array_type) else 1
        marked_num = np.sum(mark)

        if not isinstance(mark, array_type) and not isinstance(mark, (bool, np.bool_)):
            raise RuntimeError(f"Illegal mark {mark} {type(mark)}")
        elif marked_num == 0:
            raise RuntimeError(f"Out of domain! {self} {xargs} ")

        if marked_num < mark_size:
            xargs = tuple([(arg[mark] if isinstance(mark, array_type) and isinstance(arg, array_type) and arg.ndim > 0 else arg)
                          for arg in xargs])

        func = self.__functor__()

        if func is None:
            value = np.nan

        elif isinstance(func, (Functor, Expression)):
            if len(self._children) > 0:  # Traverse children
                children = []
                for child in self._children:
                    if callable(child):
                        value = child(*xargs, **kwargs)
                    elif hasattr(child, "__value__"):
                        value = child.__value__
                    elif hasattr(child, "__array__"):
                        value = child.__array__()
                    else:
                        value = child
                    children.append(value)

                if len(children) > 0:
                    xargs = tuple(children)
                    kwargs = {}

            try:
                value = func(*xargs, **kwargs)
            except Exception as error:
                raise RuntimeError(f"Error when evaluating {func} !") from error

        elif isinstance(func, numeric_type):
            value = func

        else:
            raise RuntimeError(f"Unknown functor {func} {type(func)}")

        if marked_num == mark_size:
            if not isinstance(mark, array_type):
                res = value
            elif is_scalar(value):
                res = np.full_like(mark, value, dtype=self._type_hint())
            elif isinstance(value, array_type) and value.shape == mark.shape:
                res = value
            elif value is None:
                res = None
            else:
                raise RuntimeError(f"Incorrect reuslt {self}! {value}")
        else:
            res = np.full_like(mark, self.fill_value, dtype=self._type_hint())
            res[mark] = value

        return res


    # fmt: off
    def __neg__      (self                             ) : return Expression(np.negative     ,  self     ,)
    def __add__      (self, o: NumericType | Expression) : return Expression(np.add          ,  self, o  ,)
    def __sub__      (self, o: NumericType | Expression) : return Expression(np.subtract     ,  self, o  ,)
    def __mul__      (self, o: NumericType | Expression) : return Expression(np.multiply     ,  self, o  ,)
    def __matmul__   (self, o: NumericType | Expression) : return Expression(np.matmul       ,  self, o  ,)
    def __truediv__  (self, o: NumericType | Expression) : return Expression(np.true_divide  ,  self, o  ,)
    def __pow__      (self, o: NumericType | Expression) : return Expression(np.power        ,  self, o  ,)
    def __eq__       (self, o: NumericType | Expression) : return Expression(np.equal        ,  self, o  ,)
    def __ne__       (self, o: NumericType | Expression) : return Expression(np.not_equal    ,  self, o  ,)
    def __lt__       (self, o: NumericType | Expression) : return Expression(np.less         ,  self, o  ,)
    def __le__       (self, o: NumericType | Expression) : return Expression(np.less_equal   ,  self, o  ,)
    def __gt__       (self, o: NumericType | Expression) : return Expression(np.greater      ,  self, o  ,)
    def __ge__       (self, o: NumericType | Expression) : return Expression(np.greater_equal,  self, o  ,)
    def __radd__     (self, o: NumericType | Expression) : return Expression(np.add          ,  o, self  ,)
    def __rsub__     (self, o: NumericType | Expression) : return Expression(np.subtract     ,  o, self  ,)
    def __rmul__     (self, o: NumericType | Expression) : return Expression(np.multiply     ,  o, self  ,)
    def __rmatmul__  (self, o: NumericType | Expression) : return Expression(np.matmul       ,  o, self  ,)
    def __rtruediv__ (self, o: NumericType | Expression) : return Expression(np.divide       ,  o, self  ,)
    def __rpow__     (self, o: NumericType | Expression) : return Expression(np.power        ,  o, self  ,)
    def __abs__      (self                             ) : return Expression(np.abs          ,  self     ,)
    def __pos__      (self                             ) : return Expression(np.positive     ,  self     ,)
    def __invert__   (self                             ) : return Expression(np.invert       ,  self     ,)
    def __and__      (self, o: NumericType | Expression) : return Expression(np.bitwise_and  ,  self, o  ,)
    def __or__       (self, o: NumericType | Expression) : return Expression(np.bitwise_or   ,  self, o  ,)
    def __xor__      (self, o: NumericType | Expression) : return Expression(np.bitwise_xor  ,  self, o  ,)
    def __rand__     (self, o: NumericType | Expression) : return Expression(np.bitwise_and  ,  o, self  ,)
    def __ror__      (self, o: NumericType | Expression) : return Expression(np.bitwise_or   ,  o, self  ,)
    def __rxor__     (self, o: NumericType | Expression) : return Expression(np.bitwise_xor  ,  o, self  ,)
    def __rshift__   (self, o: NumericType | Expression) : return Expression(np.right_shift  ,  self, o  ,)
    def __lshift__   (self, o: NumericType | Expression) : return Expression(np.left_shift   ,  self, o  ,)
    def __rrshift__  (self, o: NumericType | Expression) : return Expression(np.right_shift  ,  o, self  ,)
    def __rlshift__  (self, o: NumericType | Expression) : return Expression(np.left_shift   ,  o, self  ,)
    def __mod__      (self, o: NumericType | Expression) : return Expression(np.mod          ,  self, o  ,)
    def __rmod__     (self, o: NumericType | Expression) : return Expression(np.mod          ,  o, self  ,)
    def __floordiv__ (self, o: NumericType | Expression) : return Expression(np.floor_divide ,  self, o  ,)
    def __rfloordiv__(self, o: NumericType | Expression) : return Expression(np.floor_divide ,  o, self  ,)
    def __trunc__    (self                             ) : return Expression(np.trunc        ,  self     ,)
    def __round__    (self, n=None                     ) : return Expression(np.round        ,  self, n  ,)
    def __floor__    (self                             ) : return Expression(np.floor        ,  self     ,)
    def __ceil__     (self                             ) : return Expression(np.ceil         ,  self     ,)
    # fmt: on


class Variable(Expression):
    """
        Variable
        ---------
        变量是一种特殊的函数，它的值由上下文决定。
        例如：
            >>> import spdm
            >>> x = spdm.data.Variable(0,"x")
            >>> y = spdm.data.Variable(1,"y")
            >>> z = x + y
            >>> z
            <Expression   op="add" />
            >>> z(0.0, 1.0)
            1.0

    """

    def __init__(self, idx: int | str, label: str = None) -> None:
        super().__init__()
        self._idx = idx
        self._label = label if label is not None else (idx if isinstance(idx, str) else f"_{idx}")

    def __str__(self) -> str: return self._label

    @property
    def _type_hint(self) -> typing.Type:
        """ 获取函数的类型
        """
        orig_class = getattr(self, "__orig_class__", None)
        if orig_class is not None:
            return typing.get_args(orig_class)[0]
        else:
            return float

    @property
    def __label__(self): return self._label

    @property
    def index(self): return self._idx

    def __call__(self, *args, **kwargs):
        if isinstance(self._idx, str):
            return kwargs[self._idx]
        else:
            return args[self._idx]
        # if len(args) <= self._idx:
        #     raise RuntimeError(f"Variable {self} require {self._idx} args, but only {len(args)} provided!")
        # return args[self._idx]


class Piecewise(Expression):
    """ PiecewiseFunction
        ----------------
        A piecewise function. 一维或多维，分段函数
    """

    def __init__(self, func: typing.List[typing.Callable], cond: typing.List[typing.Callable], **kwargs):
        super().__init__(None, **kwargs)
        self._piecewise = (func, cond)

    def _apply(self, func, cond, x, *args, **kwargs):
        if isinstance(x, array_type):
            x = x[cond(x)]
        else:
            return func(x) if cond(x) else None

        if isinstance(func, numeric_type):
            value = np.full_like(x, func, dtype=float)
        elif callable(func):
            value = func(x)
        else:
            raise ValueError(f"PiecewiseFunction._apply() error! {func} {x}")
            # [(node(*args, **kwargs) if callable(node) else (node.__entry__().__value__() if hasattr(node, "__entry__") else node))
            #          for node in self._expr_nodes]
        return value

    def __call__(self, x, *args, **kwargs) -> NumericType:
        if isinstance(x, float):
            res = [self._apply(fun, cond, x) for fun, cond in zip(*self._piecewise) if cond(x)]
            if len(res) == 0:
                raise RuntimeError(f"Can not fit any condition! {x}")
            elif len(res) > 1:
                raise RuntimeError(f"Fit multiply condition! {x}")
            return res[0]
        elif isinstance(x, array_type):
            res = np.hstack([self._apply(fun, cond, x) for fun, cond in zip(*self._piecewise)])
            if len(res) != len(x):
                raise RuntimeError(f"PiecewiseFunction result length not equal to input length, {len(res)}!={len(x)}")
            return res
        else:
            raise TypeError(f"PiecewiseFunction only support single float or  1D array, {type(x)} {array_type}")
