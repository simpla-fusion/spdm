from __future__ import annotations

import collections.abc
import typing
import functools
import numpy as np

from ..utils.logger import logger
from ..utils.typing import ArrayType, NumericType, numeric_type, array_type
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

    def __init__(self, op: typing.Callable, *args, name=None,  **kwargs) -> None:
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

        if getattr(op, "has_children", False):
            # copy constructor
            args = op._children
            op = op._op
        elif callable(op) and len(kwargs) > 0:
            op = functools.partial(op, **kwargs)
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
        if self._name is None:
            if isinstance(op, tuple):
                op, opts, *method = op

                if len(method) > 0:
                    method = method[0]
                else:
                    method = None

                if method == "__call__" or method is None:
                    self._name = getattr(op, "__name__", op.__class__.__name__)
                else:
                    self._name = f"{op.__class__.__name__}.{method}"
            elif isinstance(op, numeric_type):
                self._name = f"[{op}]"
            else:
                self._name = getattr(op, "__name__", op.__class__.__name__)

    def __duplicate__(self) -> Expression:
        """ 复制一个新的 Expression 对象 """
        other: Expression = object.__new__(self.__class__)
        other._op = self._op
        other._name = self._name
        other._children = self._children

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
        name = getattr(ufunc, "__name__", f"{ufunc.__class__.__name__}.{method}")
        return Expression(functools.partial(getattr(ufunc, method), **kwargs), *args, name=name)

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
    def __name__(self) -> str: return self._name

    def __str__(self) -> str:
        """ 返回表达式的抽象语法树"""

        op_name = self._name

        name = _EXPR_OP_NAME.get(op_name, None)

        def _ast(v):
            if isinstance(v, Expression):
                return v.__str__()
            elif isinstance(v, np.ndarray):
                return f"<{v.shape}>"
            else:
                return str(v)
        if name is not None and len(self._children) == 2:
            return f"{_ast(self._children[0])} {name} {_ast(self._children[1])}"
        else:
            return f"{op_name}({', '.join([_ast(arg) for arg in self._children])})"

    @property
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

    @property
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

    def __domain__(self, *args) -> bool:
        """ 判断表达式是否在指定的定义域内 """
        d = [child.__domain__(*args) for child in self._children if hasattr(child, "__domain__")]
        if len(d) > 0:
            return np.bitwise_and.reduce(d)
        else:
            return True

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
        if len(args) == 0:
            return self
        elif any([isinstance(arg, Expression) for arg in args]):
            return Expression(*args, op=(self, kwargs))

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

    def compile(self, *args, ** kwargs) -> Expression:
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

    def __init__(self, idx: int | str, name: str = None) -> None:
        super().__init__(None)
        self._idx = idx
        self._name = name if name is not None else (idx if isinstance(idx, str) else f"_{idx}")

    def __str__(self) -> str: return self._name

    @ property
    def __type_hint__(self) -> typing.Type:
        """ 获取函数的类型
        """
        orig_class = getattr(self, "__orig_class__", None)
        if orig_class is not None:
            return typing.get_args(orig_class)[0]
        else:
            return float

    @ property
    def name(self): return self._name

    @ property
    def index(self): return self._idx

    def __call__(self, *args, **kwargs):
        if isinstance(self._idx, str):
            return kwargs[self._idx]
        else:
            return args[self._idx]
        # if len(args) <= self._idx:
        #     raise RuntimeError(f"Variable {self} require {self._idx} args, but only {len(args)} provided!")
        # return args[self._idx]


# _0 = Variable[float](0)
# _1 = Variable[float](1)
# _2 = Variable[float](2)
# _3 = Variable[float](3)
# _4 = Variable[float](4)
# _5 = Variable[float](5)
# _6 = Variable[float](6)
# _7 = Variable[float](7)
# _8 = Variable[float](8)
# _9 = Variable[float](9)

class Piecewise(Expression, typing.Generic[_T]):
    """ PiecewiseFunction
        ----------------
        A piecewise function. 一维或多维，分段函数
    """

    def __init__(self, func: typing.List[typing.Callable], cond: typing.List[typing.Callable], **kwargs):
        super().__init__(op=(func, cond), **kwargs)

    @ property
    def rank(self): return 1

    @ property
    def ndim(self): return 1

    def _compile(self): return self, {}

    def _apply(self, func, x, *args, **kwargs):
        if isinstance(x, array_type) and isinstance(func, numeric_type):
            value = np.full(x.shape, func)
        elif isinstance(x, numeric_type) and isinstance(func, numeric_type):
            value = func
        elif callable(func):
            value = func(x)
        else:
            raise ValueError(f"PiecewiseFunction._apply() error! {func} {x}")
            # [(node(*args, **kwargs) if callable(node) else (node.__entry__().__value__() if hasattr(node, "__entry__") else node))
            #          for node in self._expr_nodes]
        return value

    def __call__(self, x, *args, **kwargs) -> NumericType:
        if isinstance(x, float):
            res = [self._apply(fun, x) for fun, cond in zip(*self._op) if cond(x)]
            if len(res) == 0:
                raise RuntimeError(f"Can not fit any condition! {x}")
            elif len(res) > 1:
                raise RuntimeError(f"Fit multiply condition! {x}")
            return res[0]
        elif isinstance(x, array_type):
            res = np.hstack([self._apply(fun, x[cond(x)]) for fun, cond in zip(*self._op)])
            if len(res) != len(x):
                raise RuntimeError(f"PiecewiseFunction result length not equal to input length, {len(res)}!={len(x)}")
            return res
        else:
            raise TypeError(f"PiecewiseFunction only support single float or  1D array, {x}")
