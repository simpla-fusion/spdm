from __future__ import annotations

import collections.abc
import typing

import numpy as np

from ..utils.logger import logger
from ..utils.typing import ArrayType, NumericType, numeric_types
from ..utils.tags import _not_found_, _undefined_
_T = typing.TypeVar("_T")


class Expression(typing.Generic[_T]):
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

    def __init__(self,  *args,  op: typing.Callable | None = None, method: str | None = None, **kwargs) -> None:
        """
            Parameters
            ----------
            args : typing.Any
                操作数
            op : typing.Callable | None
                运算符，可以是函数，也可以是类的成员函数
            method : str | None
                运算符的成员函数名
            kwargs : typing.Any
                命名参数， 用于传递给运算符的参数

        """
        super().__init__()

        if len(args) == 1 and isinstance(args[0], Expression) and op is None:
            # copy constructor
            self._args = args[0]._args
            self._op = args[0]._op
            self._method = args[0]._method
            self._kwargs = args[0]._kwargs

            if len(kwargs) > 0:
                logger.warning(f"Ignore kwargs={kwargs}!")
        else:
            self._args = args       # 操作数
            self._op = op
            self._method = method
            self._kwargs = kwargs   # named arguments for _operator_

    def __duplicate__(self) -> Expression:
        """ 复制一个新的 Expression 对象 """
        other: Expression = object.__new__(self.__class__)
        other._args = self._args
        other._kwargs = self._kwargs
        other._op = self._op
        other._method = self._method
        return other

    def __array_ufunc__(self, ufunc, method, *args, **kwargs) -> Expression:
        """ 
            重载 numpy 的 ufunc 运算符, 用于在表达式中使用 numpy 的 ufunc 函数构建新的表达式。    
            例如：
                >>> import numpy as np
                >>> import spdm
                >>> x = spdm.data.Expression(ufunc=np.sin)
                >>> y = spdm.data.Expression(ufunc=np.cos)
                >>> z = x + y
                >>> z
                <Expression   op="add" />
                >>> z(0.0)
                1.0
        """
        return Expression(*args, op=ufunc, method=method, **kwargs)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}   op=\"{getattr(self._op,'__name__','unnamed')}\" />"

    @property
    def is_function(self) -> bool: return not self._args
    """ 判断是否为函数。 当操作数为空的时候，表达式为函数 """
    
    @property
    def __type_hint__(self) -> typing.Type:
        """ 获取函数的类型
        """
        orig_class = getattr(self, "__orig_class__", None)
        if orig_class is not None:
            return typing.get_args(orig_class)[0]
        else:
            return float

    def _apply(self, func, *args, **kwargs) -> typing.Any:
        if isinstance(func, collections.abc.Sequence) and len(func) > 0:
            return [self._apply(f, *args, **kwargs) for f in func]
        elif hasattr(func, "__entry__"):
            return self._apply(func.__entry__().__value__(), *args, **kwargs)
        elif callable(func):
            return func(*args, **kwargs)
        elif isinstance(func, numeric_types):
            return func
        else:
            if func:
                logger.warning(f"Ignore illegal function {func}!")
            return args

    def __call__(self,  *args: NumericType) -> ArrayType:
        """
            重载函数调用运算符，用于计算表达式的值

            TODO:
            - support JIT compilation
            - support broadcasting?
            - support multiple meshes?
        """

        value = self._apply(self._args, *args)

        res = _undefined_

        if self._method is not None:  # operator is  member function of self._op
            method = getattr(self._op, self._method, None)
            if method is not None:
                res = method(*value, **self._kwargs)
            else:
                raise AttributeError(f"{self._op.__class__.__name__} has not method {self._method}!")
        elif callable(self._op):  # operator is a function
            res = self._op(*value, **self._kwargs)
        elif not self._op:  # constant Expression
            res = args if len(args) != 1 else args[0]

        if res is _undefined_:
            raise AttributeError(f"Cannot apply {self._op} to {value}!")

        return res


    # fmt: off
    def __neg__      (self                             ) : return Expression(self     , op=np.negative     )
    def __add__      (self, o: NumericType | Expression) : return Expression(self, o  , op=np.add          )
    def __sub__      (self, o: NumericType | Expression) : return Expression(self, o  , op=np.subtract     )
    def __mul__      (self, o: NumericType | Expression) : return Expression(self, o  , op=np.multiply     )
    def __matmul__   (self, o: NumericType | Expression) : return Expression(self, o  , op=np.matmul       )
    def __truediv__  (self, o: NumericType | Expression) : return Expression(self, o  , op=np.true_divide  )
    def __pow__      (self, o: NumericType | Expression) : return Expression(self, o  , op=np.power        )
    def __eq__       (self, o: NumericType | Expression) : return Expression(self, o  , op=np.equal        )
    def __ne__       (self, o: NumericType | Expression) : return Expression(self, o  , op=np.not_equal    )
    def __lt__       (self, o: NumericType | Expression) : return Expression(self, o  , op=np.less         )
    def __le__       (self, o: NumericType | Expression) : return Expression(self, o  , op=np.less_equal   )
    def __gt__       (self, o: NumericType | Expression) : return Expression(self, o  , op=np.greater      )
    def __ge__       (self, o: NumericType | Expression) : return Expression(self, o  , op=np.greater_equal)
    def __radd__     (self, o: NumericType | Expression) : return Expression(o, self  , op=np.add          )
    def __rsub__     (self, o: NumericType | Expression) : return Expression(o, self  , op=np.subtract     )
    def __rmul__     (self, o: NumericType | Expression) : return Expression(o, self  , op=np.multiply     )
    def __rmatmul__  (self, o: NumericType | Expression) : return Expression(o, self  , op=np.matmul       )
    def __rtruediv__ (self, o: NumericType | Expression) : return Expression(o, self  , op=np.divide       )
    def __rpow__     (self, o: NumericType | Expression) : return Expression(o, self  , op=np.power        )
    def __abs__      (self                             ) : return Expression(self     , op=np.abs          )
    def __pos__      (self                             ) : return Expression(self     , op=np.positive     )
    def __invert__   (self                             ) : return Expression(self     , op=np.invert       )
    def __and__      (self, o: NumericType | Expression) : return Expression(self, o  , op=np.bitwise_and  )
    def __or__       (self, o: NumericType | Expression) : return Expression(self, o  , op=np.bitwise_or   )
    def __xor__      (self, o: NumericType | Expression) : return Expression(self, o  , op=np.bitwise_xor  )
    def __rand__     (self, o: NumericType | Expression) : return Expression(o, self  , op=np.bitwise_and  )
    def __ror__      (self, o: NumericType | Expression) : return Expression(o, self  , op=np.bitwise_or   )
    def __rxor__     (self, o: NumericType | Expression) : return Expression(o, self  , op=np.bitwise_xor  )
    def __rshift__   (self, o: NumericType | Expression) : return Expression(self, o  , op=np.right_shift  )
    def __lshift__   (self, o: NumericType | Expression) : return Expression(self, o  , op=np.left_shift   )
    def __rrshift__  (self, o: NumericType | Expression) : return Expression(o, self  , op=np.right_shift  )
    def __rlshift__  (self, o: NumericType | Expression) : return Expression(o, self  , op=np.left_shift   )
    def __mod__      (self, o: NumericType | Expression) : return Expression(self, o  , op=np.mod          )
    def __rmod__     (self, o: NumericType | Expression) : return Expression(o, self  , op=np.mod          )
    def __floordiv__ (self, o: NumericType | Expression) : return Expression(self, o  , op=np.floor_divide )
    def __rfloordiv__(self, o: NumericType | Expression) : return Expression(o, self  , op=np.floor_divide )
    def __trunc__    (self                             ) : return Expression(self     , op=np.trunc        )
    def __round__    (self, n=None                     ) : return Expression(self, n  , op=np.round        )
    def __floor__    (self                             ) : return Expression(self     , op=np.floor        )
    def __ceil__     (self                             ) : return Expression(self     , op=np.ceil         )
    # fmt: on


class Variable(Expression[_T]):
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

    def __init__(self, idx: int, name: str = None) -> None:
        super().__init__(op=lambda *args, _idx=idx: args[_idx])
        self._name = name if name is not None else f"x_{idx}"
        self._idx = idx

    @property
    def name(self): return self._name

    @property
    def index(self): return self._idx


_0 = Variable(0)
_1 = Variable(1)
_2 = Variable(2)
_3 = Variable(3)
_4 = Variable(4)
_5 = Variable(5)
_6 = Variable(6)
_7 = Variable(7)
_8 = Variable(8)
_9 = Variable(9)
