from __future__ import annotations

import collections.abc
import typing
from functools import cached_property
import numpy as np

from ..utils.logger import logger
from ..utils.typing import ArrayType, NumericType, numeric_type
from ..utils.tags import _not_found_, _undefined_


_EXPR_OP_NAME = {
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

    def __init__(self,  *args,  op: typing.Callable | typing.Tuple[typing.Callable, str, typing.Dict[str, typing.Any]] = None, name=None,  **kwargs) -> None:
        """
            Parameters
            ----------
            args : typing.Any
                操作数
            op : typing.Callable  | typing.Tuple[typing.Callable, str, typing.Dict[str, typing.Any]] 
                运算符，可以是函数，也可以是类的成员函数 

            kwargs : typing.Any
                命名参数， 用于传递给运算符的参数

        """
        super().__init__()

        if op is None and len(args) == 1:  # copy constructor
            if getattr(args[0], "has_children", False):
                # copy constructor
                op = args[0]._op
                args = args[0]._children
            elif callable(args[0]):
                op = (args[0], kwargs)
                args = ()
                kwargs = {}

        self._children = args
        self._op = op

        if len(kwargs) == 0:
            pass
        elif isinstance(self._op, tuple):  # require (op,opts,method)
            self._op[1].update(kwargs)
        elif callable(self._op):
            self._op = (self._op, kwargs)
        else:
            logger.warning(f"Ignore kwargs={kwargs}! op={self._op}")

        self._name = name

    def __duplicate__(self) -> Expression:
        """ 复制一个新的 Expression 对象 """
        other: Expression = object.__new__(self.__class__)
        other._children = self._children
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
        return Expression(*args, op=(ufunc, kwargs, method))

    def __array__(self, *args) -> ArrayType:
        raise NotImplementedError(f"__array__({args}) is not implemented!")

    @property
    def has_children(self) -> bool: return len(self._children) > 0
    """ 判断是否有子节点"""

    @property
    def empty(self) -> bool: return not self.has_children and self._op is None

    @property
    def callable(self): return self._op is not None or self.has_children

    # @property
    # def is_function(self) -> bool: return not self.has_children and self._op is not None

    @property
    def op_name(self) -> str:
        if isinstance(self._op, tuple):
            op, opts, *method = self._op

            if len(method) > 0:
                method = method[0]
            else:
                method = None

            if method == "__call__" or method is None:
                return getattr(op, "__name__", op.__class__.__name__)
            else:
                return f"{op.__class__.__name__}.{method}"
        elif isinstance(self._op, numeric_type):
            return f"[{self._op}]"
        else:
            return getattr(self._op, "__name__", self._op.__class__.__name__)

    def __str__(self) -> str:
        """ 返回表达式的抽象语法树"""
        if self._name is not None:
            return self._name

        op_name = self.op_name

        _name = _EXPR_OP_NAME.get(op_name, None)

        def _ast(v):
            if isinstance(v, Expression):
                return v.__str__()
            elif isinstance(v, np.ndarray):
                return f"<{v.shape}>"
            else:
                return str(v)
        if _name is not None and len(self._children) == 2:
            return f"{_ast(self._children[0])} {_name} {_ast(self._children[1])}"
        else:
            return f"{op_name}({', '.join([_ast(arg) for arg in self._children])})"

    @cached_property
    def __type_hint__(self):
        """ 获取表达式的类型
        """
        def get_type(obj):
            if not isinstance(obj, Expression):
                return set([None])
            elif len(obj._children) > 0:
                return set([t for o in obj._children for t in get_type(o) if hasattr(t, "mesh")])
            else:
                return set([obj.__class__])
        tp = list(get_type(self))
        if len(tp) == 0:
            return float
        elif len(tp) == 1:
            return tp[0]
        else:
            raise TypeError(f"Can not determint the type of expresion {self}! {tp}")

    @cached_property
    def __mesh__(self):
        """ 获取表达式的最大有效定义域 """
        def get_mesh(obj):
            if not isinstance(obj, Expression):
                return [None]
            elif len(obj._children) > 0:
                return [d for o in obj._children for d in get_mesh(o) if d is not None]
            else:
                return [getattr(obj, "mesh", None)]
        new_list = []
        for d in get_mesh(self):
            if d not in new_list:
                new_list.append(d)
        if len(new_list) == 0:
            raise RuntimeError(f"Expression can not reslove mesh! mesh list={new_list}")
        elif len(new_list) != 1:
            logger.warning(f"get ({len(new_list)}) results, only take the first! ")

        return new_list[0]

    def _fetch_children(self, *args, **kwargs) -> typing.List[typing.Any]:
        if len(self._children) == 0:
            return args, kwargs

        children = []

        for child in self._children:

            if callable(child):
                value = child(*args, **kwargs)
            elif hasattr(child, "__value__"):
                value = child.__value__()
            elif hasattr(child, "__array__"):
                value = child.__array__()
            else:
                value = child
            children.append(value)

        return children, {}

    def _fetch_op(self): return self._op

    def __call__(self, *args, **kwargs) -> ArrayType | Expression:
        """
            重载函数调用运算符，用于计算表达式的值

            TODO:
            - support JIT compilation
            - support broadcasting?
            - support multiple meshes?
        """
        if any([isinstance(arg, Expression) for arg in args]):
            if self.has_children:
                return self.__class__(*args, op=(self, kwargs))
            else:
                other = self.__duplicate__()
                other._children = args
                return other

        # TODO: 对 expression 执行进行计数

        args, kwargs = self._fetch_children(*args, **kwargs)

        op = self._fetch_op()

        if isinstance(op, tuple):
            op, opts, *method = op
            if len(method) > 0 and isinstance(method[0], str):
                op = getattr(op, method[0])
            if opts is not None and len(opts) > 0:
                kwargs.update(opts)

        res = _undefined_

        if not op:  # tuple Expression
            res = args if len(args) != 1 else args[0]
        elif not callable(op):
            raise RuntimeError(f"Unknown op={op} children={self._children}!")
        else:
            try:
                res = op(*args, **kwargs)
            except Exception as error:
                raise RuntimeError(
                    f"Error when apply {self.__str__()} op={op} args={args} kwargs={kwargs}!") from error

        if res is _undefined_:
            raise RuntimeError(f"Unknown op={op} expr={self._children}!")

        return res

    def _compile(self, *args, ** kwargs) -> Expression:
        """ 编译函数，返回一个新的(加速的)函数对象 
            TODO：
                - JIT compile
        """
        if len(args) > 0:
            # TODO: 支持自动微分，auto-grad?
            raise NotImplementedError(f"TODO: derivative/antiderivative args={args} !")

        f_class = self.__type_hint__

        f_mesh = self.__mesh__

        points = getattr(self, "points", None)

        if points is None:  # for Field
            points = getattr(f_mesh, "points")

        if points is not None:
            pass
        elif isinstance(f_mesh, tuple):  # rectlinear mesh for Function
            points = np.meshgrid(*f_mesh)
        else:
            raise RuntimeError(f"Can not reslove mesh {f_mesh}!")

        return f_class(self.__call__(*points), mesh=f_mesh, name=f"[{self.__str__()}]", **kwargs)

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
        super().__init__()
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

    def __call__(self, *args, **kwargs):
        if len(args) <= self._idx:
            raise RuntimeError(f"Variable {self} require {self._idx} args, but only {len(args)} provided!")
        return args[self._idx]


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
