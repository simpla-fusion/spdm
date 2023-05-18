from __future__ import annotations

import collections.abc
import typing
from functools import cached_property
import numpy as np

from ..utils.logger import logger
from ..utils.typing import ArrayType, NumericType, numeric_types
from ..utils.tags import _not_found_, _undefined_


class Expression(object):
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

    def __init__(self,  *args,  op: typing.Callable | typing.Tuple[typing.Callable, str, typing.Dict[str, typing.Any]] = None, method=None, **kwargs) -> None:
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
            self._expr_nodes = args[0]._expr_nodes
            self._op = args[0]._op
        else:
            self._expr_nodes = args       # 操作数
            if method is not None or len(kwargs) > 0:
                self._op = (op, method, kwargs)   # named arguments for _operator_
                kwargs = {}
            else:
                self._op = op

        if len(kwargs) > 0:
            logger.warning(f"Ignore kwargs={kwargs}!")

    def __duplicate__(self) -> Expression:
        """ 复制一个新的 Expression 对象 """
        other: Expression = object.__new__(self.__class__)
        other._expr_nodes = self._expr_nodes
        other._op = self._op
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
        return Expression(*args, op=(ufunc, method, kwargs))

    @property
    def is_function(self) -> bool: return not self._expr_nodes
    """ 判断是否为函数。 当操作数为空的时候，表达式为函数 """

    OP_NAME = {
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
        "greater_equa": ">=",
        "add": "+",
        "subtract": "-",
        "multiply": "*",
        "matmul": "*",
        "divide": "/",
        "power": "^",
        # "abs": "",
        "positive": "+",
        # "invert": "",
        # "bitwise_and": "",
        # "bitwise_or": "",
        # "bitwise_xor": "",
        # "bitwise_and": "",
        # "bitwise_or": "",
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

    @property
    def op_name(self) -> str:
        if isinstance(self._op, tuple):
            op, method, _ = self._op
            if method == "__call__" or method is None:
                return getattr(op, "__name__", op.__class__.__name__)
            else:
                return f"{op.__class__.__name__}.{method}"
        else:
            return getattr(self._op, "__name__", self._op.__class__.__name__)

    def __str__(self) -> str:
        """ 返回表达式的抽象语法树"""
        op_name = self.op_name

        _name = Expression.OP_NAME.get(op_name, None)

        def _ast(v):
            if isinstance(v, Expression):
                return v.__str__()
            elif isinstance(v, np.ndarray):
                return f"<{v.shape}>"
            else:
                return str(v)
        if _name is not None and len(self._expr_nodes) == 2:
            return f"{_ast(self._expr_nodes[0])} {_name} {_ast(self._expr_nodes[1])}"
        else:
            return f"{op_name}({', '.join([_ast(arg) for arg in self._expr_nodes])})"

    def __call__(self,  *args: NumericType, **kwargs) -> ArrayType:
        """
            重载函数调用运算符，用于计算表达式的值

            TODO:
            - support JIT compilation
            - support broadcasting?
            - support multiple meshes?
        """
        try:
            res = self._eval(*args, **kwargs)
        except Exception as error:
            raise RuntimeError(f"Error when eval {self} with args={len(args)} and kwargs={kwargs}!") from error
        return res

    def _eval(self, *args, **kwargs) -> ArrayType:

        if len(self._expr_nodes) > 0:
            args = [(node(*args, **kwargs) if callable(node) else (node.__entry__().__value__() if hasattr(node, "__entry__") else node))
                    for node in self._expr_nodes]

        res = _undefined_

        if isinstance(self._op, tuple):
            op, method, opts = self._op
            if isinstance(method, str):
                op = getattr(op, method, None)
                if op is None or op is _not_found_:
                    raise AttributeError(f"{self._op[0].__class__.__name__} has not method {method}!")
        else:
            op = self._op
            opts = {}

        if callable(op):  # operator is a function
            try:
                res = op(*args, ** opts)
                # TODO: 对 expression 执行进行计数
            except Exception as error:
                raise RuntimeError(f"Error when apply {self.__str__()} !") from error

        elif not op:  # constant Expression
            res = args if len(args) != 1 else args[0]

        if res is _undefined_:
            raise AttributeError(f"Cannot apply {self._op} to {args}!")

        return res

    def is_function(self) -> bool:
        def _is_function(obj):
            return isinstance(obj, Expression) and (hasattr(self, 'dims') or any([_is_function(o) for o in obj._expr_nodes]))
        return _is_function(self)

    @property
    def is_field(self) -> bool:
        def _is_field(obj):
            return isinstance(obj, Expression) and (hasattr(self, 'mesh') or any([_is_field(o) for o in obj._expr_nodes]))
        return _is_field(self)

    @cached_property
    def __type_hint__(self):
        """ 获取表达式的类型
        """
        def get_type(obj):
            if not isinstance(obj, Expression):
                return set([None])
            elif len(obj._expr_nodes) > 0:
                return set([t for o in obj._expr_nodes for t in get_type(o) if hasattr(t, "domain")])
            else:
                return set([obj.__class__])
        tp = list(get_type(self))
        if len(tp) != 1:
            raise TypeError(f"Can not determint the type of expresion {self}! {tp}")
        return tp[0]

    @cached_property
    def __domain__(self):
        """ 获取表达式的最大有效定义域 """
        def get_domain(obj):
            if not isinstance(obj, Expression):
                return [None]
            elif len(obj._expr_nodes) > 0:
                return [d for o in obj._expr_nodes for d in get_domain(o) if d is not None]
            else:
                return [getattr(obj, "domain", None)]
        new_list = []
        for d in get_domain(self):
            if d not in new_list:
                new_list.append(d)
        if len(new_list) == 0:
            raise RuntimeError(f"Expression can not reslove domain! domain list={new_list}")
        elif len(new_list) != 1:
            logger.warning(f"get ({len(new_list)}) results, only take the first! ")

        return new_list[0]

    def compile(self, *args, **kwargs) -> Expression:
        """ 编译函数，返回一个新的(加速的)函数对象 
            TODO：
                - JIT compile
        """
        f_class = self.__type_hint__

        domain = self.__domain__

        if domain is None:
            domain = args
        if isinstance(domain, tuple):
            return f_class(self.__call__(*domain), *domain, name=f"[{self.__str__()}]", **kwargs)
        else:
            return f_class(self.__call__(*domain.points),  domain=domain, name=f"[{self.__str__()}]", **kwargs)

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


_T = typing.TypeVar("_T")


class Variable(Expression, typing.Generic[_T]):
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

    def __str__(self) -> str: return self._name

    @property
    def __type_hint__(self) -> typing.Type:
        """ 获取函数的类型
        """
        orig_class = getattr(self, "__orig_class__", None)
        if orig_class is not None:
            return typing.get_args(orig_class)[0]
        else:
            return float

    @property
    def name(self): return self._name

    @property
    def index(self): return self._idx


_0 = Variable[float](0)
_1 = Variable[float](1)
_2 = Variable[float](2)
_3 = Variable[float](3)
_4 = Variable[float](4)
_5 = Variable[float](5)
_6 = Variable[float](6)
_7 = Variable[float](7)
_8 = Variable[float](8)
_9 = Variable[float](9)
