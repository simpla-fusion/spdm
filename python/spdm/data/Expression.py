from __future__ import annotations
import warnings
import typing
from copy import copy, deepcopy
import functools
import collections.abc
import numpy as np

from ..utils.typing import ArrayType
from ..utils.tags import _not_found_
from ..utils.typing import ArrayType, NumericType, array_type, as_array, is_scalar, is_array, numeric_type
from ..utils.logger import logger
from ..numlib.interpolate import interpolate

from .Functor import Functor
from .Entry import Entry
from .HTree import HTreeNode, HTree, HTreeNode
from .Domain import DomainBase
from .Path import update_tree, Path
from .Functor import Functor, DerivativeOp


_T = typing.TypeVar("_T", float, bool, array_type, HTreeNode)


class Expression(HTreeNode):
    """Expression

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

    def __init__(self, op, *args, **kwargs) -> None:
        """
        Parameters


        op : typing.Callable  | ExprOp
            运算符，可以是函数，也可以是类的成员函数
        args : typing.Any
            操作数
        kwargs : typing.Any
            命名参数， 用于传递给运算符的参数

        """

        if self.__class__ is not Expression or callable(op) or op is None:
            super().__init__(**kwargs)

            self._op = op
            self._children = args

        else:
            if is_scalar(op):
                match op:
                    case 0:
                        TP = ConstantZero
                    case 1:
                        TP = ConstantOne
                    case _:
                        TP = Scalar

            elif isinstance(op, array_type):
                # 构建插值函数
                from .Function import Function

                TP = Function

            else:
                return
                # raise RuntimeError(f"Unknown functor {op} {type(op)}")

            self.__class__ = TP

            TP.__init__(self, op, *args, **kwargs)

    def __copy__(self) -> Expression:
        """复制一个新的 Expression 对象"""
        other: Expression = super().__copy__()
        other._op = copy(self._op)
        other._children = copy(self._children)
        return other

    def __serialize__(self, dumper=None):
        logger.debug(f"TODO: __serialize__ {self}")
        return None

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
        if method != "__call__" or len(kwargs) > 0:
            return Expression(Functor(ufunc, method=method, **kwargs), *args)
        else:
            return Expression(ufunc, *args)

    def __array__(self) -> ArrayType:
        """在定义域上计算表达式。"""
        if not is_array(self._cache) and self.domain is not None:
            self._cache = self.__call__(*self.domain.points)
        if not isinstance(self._cache, (np.ndarray, float, int, bool)):
            raise RuntimeError(f"Can not calcuate! {self._cache}")

        return self._cache

    @property
    def domain(self) -> DomainBase | None:
        """返回表达式的定义域"""

        for child in self._children:
            domain = getattr(child, "domain", None)
            if domain is not None:
                break
        else:
            domain = None

        return domain

    @property
    def has_children(self) -> bool:
        """判断是否有子节点"""
        return len(self._children) > 0

    @property
    def empty(self) -> bool:
        return not self.has_children and self._op is None

    @property
    def callable(self):
        return callable(self._op) or self.has_children

    @property
    def __label__(self) -> str:
        label = self._metadata.get("label", _not_found_)
        if label is _not_found_:
            label = self._render_latex_()
        return str(label)

    def __str__(self) -> str:
        return self.__label__ or self.__name__

    def __repr__(self) -> str:
        return self._render_latex_()

    def _render_latex_(self, show=False) -> str:
        vargs = []
        for expr in self._children:
            if isinstance(expr, (bool, int, float, complex)):
                res = str(expr)
            elif expr is None:
                res = "n.a"
            elif isinstance(expr, np.ndarray):
                if len(expr.shape) == 0:
                    res = f"{expr.item()}"
                else:
                    res = f"{expr.dtype}[{expr.shape}]"
            elif isinstance(expr, Variable):
                res = f"{{{expr.__label__}}}"
            elif isinstance(expr, Expression):
                res = expr._render_latex_()
            elif isinstance(expr, np.ufunc):
                res = expr.__name__
            else:
                res = expr.__class__.__name__
            vargs.append(res)

        if (op_tag := self._metadata.get("label", self._metadata.get("name", None))) is not None:
            if len(vargs) == 0:
                res = op_tag
            else:
                res: str = rf"{op_tag}\left({','.join(vargs)}\right)"

        elif isinstance(self._op, np.ufunc):
            op_tag = EXPR_OP_TAG.get(self._op.__name__, self._op.__name__)

            if self._op.nin == 1:
                res = rf"{op_tag}{{{vargs[0]}}}"

            elif self._op.nin == 2:
                if op_tag == "/":
                    res = f"\\frac{{{vargs[0]}}}{{{vargs[1]}}}"
                else:
                    res = rf"{{{vargs[0]}}} {op_tag} {{{vargs[1]}}}"
            else:
                raise RuntimeError(f"Tri-op is not defined!")

        else:
            if isinstance(self._op, Expression):
                op_tag = self._op.__label__
            else:
                op_tag = self._op.__class__.__name__

            res: str = rf"{op_tag}\left({','.join(vargs)}\right)"

        if show:
            return rf"$${res}$$"
        else:
            return res

    def _repr_latex_(self) -> str:
        """for jupyter notebook display"""
        # label = self._metadata.get("label", None) or self.name
        # if label is not None or self._op is None:
        #     return rf"$${label}$$"
        # else:
        return self._render_latex_(show=True)

    @property
    def dtype(self):
        return self._type_hint()

    def _type_hint(self, *args):
        """TODO:获取表达式的类型"""
        return float

    def __eval__(self, *args, **kwargs) -> typing.Any:
        if self._op is None:
            res = np.nan

        elif callable(self._op):
            res = np.nan

            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                try:
                    res = self._op(*args, **kwargs)
                except RuntimeWarning as error:
                    logger.exception(f"{self._render_latex_()} ,{args} ")

        elif isinstance(self._op, numeric_type):
            res = self._op

        else:
            raise RuntimeError(f"Unknown functor { self._op} {type( self._op)}")

        return res

    def __call__(self, *args: _T, **kwargs) -> _T:
        """
        重载函数调用运算符，用于计算表达式的值

        TODO:
        - support JIT compilation
        - support broadcasting?
        - support multiple meshes?

        Parameters

        xargs : NumericType
            自变量/坐标，可以是标量，也可以是数组
        kwargs : typing.Any
            命名参数，用于传递给运算符的参数
        """
        if len(args) == 0:
            if len(kwargs) > 0:
                logger.warning(f"ignore {kwargs}")
            return self

        elif any([callable(val) for val in args]):
            return Expression(self, *args, **kwargs)

        else:
            try:
                if len(self._children) > 0:
                    args = [(child(*args, **kwargs) if callable(child) else child) for child in self._children]
                    kwargs = {}
                res = self.__eval__(*args)
            except Exception as error:
                raise RuntimeError(f"Failure to calculate  equation {self._repr_latex_()} !") from error

        return res

    def fetch(self, *args, _parent=None, **kwargs):
        res = self.__call__(*args, **kwargs)

        if res is self:
            res = self.__copy__()

        if isinstance(res, HTreeNode):
            res._parent = _parent

        return res

    def integral(self, *args, **kwargs) -> float:
        raise NotImplementedError()

    def derivative(self, order: int, *args, **kwargs) -> Derivative | Expression:
        return Derivative(order, self, *args, **kwargs) if order != 0 else self

    def antiderivative(self, order: int, *args, **kwargs) -> Antiderivative | Expression:
        return Antiderivative(order, self, *args, **kwargs) if order != 0 else self

    def partial_derivative(self, order: typing.Tuple[int, ...], *args, **kwargs) -> PartialDerivative | Expression:
        return PartialDerivative(order, self, *args, **kwargs) if order is not None and len(order) > 0 else self

    def pd(self, *order, **kwargs) -> PartialDerivative | Expression:
        return self.partial_derivative(order, **kwargs)

    @property
    def d(self) -> Expression:
        """1st derivative 一阶导数"""
        return self.derivative(1)

    @property
    def d2(self) -> Expression:
        """2nd derivative 二阶导数"""
        return self.derivative(2)

    @property
    def I(self) -> Expression:
        """antiderivative 原函数"""
        return self.antiderivative(1)

    @property
    def dln(self) -> Expression:
        """logarithmic derivative 对数求导"""
        return self.derivative(1) / self

    def find_roots(self, *args, **kwargs) -> typing.Generator[float, None, None]:
        raise NotImplementedError(f"TODO: find_roots")

    # fmt: off
    def __neg__      (self                             ) : return Expression(np.negative     ,  self     )
    def __add__      (self, o: NumericType | Expression) : return Expression(np.add          ,  self, o  ) if not ((is_scalar(o) and o == 0 ) or isinstance(o, ConstantZero) or o is _not_found_ and o is None) else self
    def __sub__      (self, o: NumericType | Expression) : return Expression(np.subtract     ,  self, o  ) if not ((is_scalar(o) and o == 0 ) or isinstance(o, ConstantZero) or o is _not_found_ and o is None) else self
    def __mul__      (self, o: NumericType | Expression) : return Expression(np.multiply     ,  self, o  ) if not (is_scalar(o) and (o ==0 or o==1)) else (ConstantZero() if o==0 else self)
    def __matmul__   (self, o: NumericType | Expression) : return Expression(np.matmul       ,  self, o  ) if not (is_scalar(o) and (o ==0 or o==1)) else (ConstantZero() if o==0 else self)
    def __truediv__  (self, o: NumericType | Expression) : return Expression(np.true_divide  ,  self, o  ) if not (is_scalar(o) and (o ==0 or o==1)) else (Scalar(np.nan) if o==0 else self)
    def __pow__      (self, o: NumericType | Expression) : return Expression(np.power        ,  self, o  ) if not (is_scalar(o) and (o ==0 or o==1)) else (ConstantOne() if o==0 else self)
    def __eq__       (self, o: NumericType | Expression) : return Expression(np.equal        ,  self, o  )
    def __ne__       (self, o: NumericType | Expression) : return Expression(np.not_equal    ,  self, o  )
    def __lt__       (self, o: NumericType | Expression) : return Expression(np.less         ,  self, o  )
    def __le__       (self, o: NumericType | Expression) : return Expression(np.less_equal   ,  self, o  )
    def __gt__       (self, o: NumericType | Expression) : return Expression(np.greater      ,  self, o  )
    def __ge__       (self, o: NumericType | Expression) : return Expression(np.greater_equal,  self, o  )
    def __radd__     (self, o: NumericType | Expression) : return Expression(np.add          ,  o, self  ) if not ((is_scalar(o) and o == 0 ) or isinstance(o, ConstantZero) or o is _not_found_ and o is None) else self
    def __rsub__     (self, o: NumericType | Expression) : return Expression(np.subtract     ,  o, self  ) if not ((is_scalar(o) and o == 0 ) or isinstance(o, ConstantZero) or o is _not_found_ and o is None) else self.__neg__()
    def __rmul__     (self, o: NumericType | Expression) : return Expression(np.multiply     ,  o, self  ) if not (is_scalar(o) and (o ==0 or o==1)) else (ConstantZero() if o==0 else self)
    def __rmatmul__  (self, o: NumericType | Expression) : return Expression(np.matmul       ,  o, self  ) if not (is_scalar(o) and (o ==0 or o==1)) else (ConstantZero() if o==0 else self)
    def __rtruediv__ (self, o: NumericType | Expression) : return Expression(np.divide       ,  o, self  )
    def __rpow__     (self, o: NumericType | Expression) : return Expression(np.power        ,  o, self  ) if not (is_scalar(o) and o ==1)  else ConstantOne()
    def __abs__      (self                             ) : return Expression(np.abs          ,  self     )
    def __pos__      (self                             ) : return Expression(np.positive     ,  self     )
    def __invert__   (self                             ) : return Expression(np.invert       ,  self     )
    def __and__      (self, o: NumericType | Expression) : return Expression(np.bitwise_and  ,  self, o  ) if not isinstance(o,bool) else ( self if o ==True else False)
    def __or__       (self, o: NumericType | Expression) : return Expression(np.bitwise_or   ,  self, o  ) if not isinstance(o,bool) else ( True if o ==True else self)
    def __xor__      (self, o: NumericType | Expression) : return Expression(np.bitwise_xor  ,  self, o  )
    def __rand__     (self, o: NumericType | Expression) : return Expression(np.bitwise_and  ,  o, self  ) if not isinstance(o,bool) else ( self if o ==True else False)
    def __ror__      (self, o: NumericType | Expression) : return Expression(np.bitwise_or   ,  o, self  ) if not isinstance(o,bool) else ( True if o ==True else self)
    def __rxor__     (self, o: NumericType | Expression) : return Expression(np.bitwise_xor  ,  o, self  )
    def __rshift__   (self, o: NumericType | Expression) : return Expression(np.right_shift  ,  self, o  )
    def __lshift__   (self, o: NumericType | Expression) : return Expression(np.left_shift   ,  self, o  )
    def __rrshift__  (self, o: NumericType | Expression) : return Expression(np.right_shift  ,  o, self  )
    def __rlshift__  (self, o: NumericType | Expression) : return Expression(np.left_shift   ,  o, self  )
    def __mod__      (self, o: NumericType | Expression) : return Expression(np.mod          ,  self, o  )
    def __rmod__     (self, o: NumericType | Expression) : return Expression(np.mod          ,  o, self  )
    def __floordiv__ (self, o: NumericType | Expression) : return Expression(np.floor_divide ,  self, o  )
    def __rfloordiv__(self, o: NumericType | Expression) : return Expression(np.floor_divide ,  o, self  )
    def __trunc__    (self                             ) : return Expression(np.trunc        ,  self     )
    def __round__    (self, n=None                     ) : return Expression(np.round        ,  self, n  )
    def __floor__    (self                             ) : return Expression(np.floor        ,  self     )
    def __ceil__     (self                             ) : return Expression(np.ceil         ,  self     )
    # fmt: on


EXPR_OP_TAG = {
    "negative": "-",
    "add": "+",
    "subtract": "-",
    "multiply": r"\cdot",
    "matmul": r"\cdot",
    "true_divide": "/",
    "power": "^",
    "equal": "==",
    "not_equal": "!",
    "less": "<",
    "less_equal": "<=",
    "greater": ">",
    "greater_equal": ">=",
    "add": "+",
    "subtract": "-",
    "multiply": r"\cdot",
    "matmul": r"\cdot",
    "divide": "/",
    "power": "^",
    # "abs": "",
    "positive": "+",
    # "invert": "",
    "bitwise_and": "&",
    "bitwise_or": "|",
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
    "sqrt": r"\sqrt",
}


class Variable(Expression):
    """Variable

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

    def __init__(self, idx: int | str, name: str = None, **kwargs) -> None:
        if name is None:
            name = idx if isinstance(idx, str) else f"_{idx}"
        super().__init__(None, name=name, **kwargs)
        self._idx = idx

    def __copy__(self) -> Scalar:
        res = super().__copy__()
        res._idx = self._idx
        return res

    @property
    def _type_hint(self) -> typing.Type:
        """获取函数的类型"""
        orig_class = getattr(self, "__orig_class__", None)
        if orig_class is not None:
            return typing.get_args(orig_class)[0]
        else:
            return float

    @property
    def index(self):
        return self._idx

    def __call__(self, *args, **kwargs):
        if all([isinstance(a, Variable) for a in args]):
            res = self
        elif len(args) == 1 and isinstance(args[0], HTree):
            pth = Path(self.__name__)
            res = pth.get(args[0], _not_found_)
            if res is _not_found_:
                res = pth.get(kwargs, _not_found_)
            if res is _not_found_:
                raise RuntimeError(f"Can not get variable {self.__name__}")
        elif self._idx < len(args):
            res = args[self._idx]
        elif self.__name__.isidentifier():
            res = kwargs.get(self.__name__)
        else:
            raise RuntimeError(
                f"Variable {self.__label__} require {self._idx+1} args, or {self.__name__} in kwargs but only {args} provided!"
            )

        return res

    def __repr__(self) -> str:
        return self.__label__


class Scalar(Expression):
    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 0 or not isinstance(args[0], (float, int, bool)):
            raise ValueError(f"args should be float|int|bool")
        super().__init__(None, *args, **kwargs)

    @property
    def __label__(self):
        return self._children[0]

    def __array__(self) -> ArrayType:
        return np.array(self._children[0])

    def __str__(self):
        return str(self._children[0])

    def __repr__(self) -> str:
        return str(self._children[0])

    def __equal__(self, other) -> bool:
        return self._children[0] == other

    def __call__(self, *args, **kwargs):
        return self._children[0]

    def derivative(self, *args, **kwargs):
        return ConstantZero(_parent=self._parent, **kwargs)


class ConstantZero(Scalar):
    def __init__(self, *args, **kwargs):
        super().__init__(0, **kwargs)

    # fmt: off
    def __neg__      (self                             ) : return self
    def __add__      (self, o: NumericType | Expression) : return o
    def __sub__      (self, o: NumericType | Expression) : return Expression(np.negative     ,  o  ) 
    def __mul__      (self, o: NumericType | Expression) : return self
    def __matmul__   (self, o: NumericType | Expression) : return self
    def __truediv__  (self, o: NumericType | Expression) : return self
    def __pow__      (self, o: NumericType | Expression) : return self
    def __radd__     (self, o: NumericType | Expression) : return o
    def __rsub__     (self, o: NumericType | Expression) : return o
    def __rmul__     (self, o: NumericType | Expression) : return self
    def __rmatmul__  (self, o: NumericType | Expression) : return self
    def __rtruediv__ (self, o: NumericType | Expression) : return Scalar(np.nan)
    def __rpow__     (self, o: NumericType | Expression) : return one
    def __abs__      (self                             ) : return self
    # fmt: on


class ConstantOne(Scalar):
    def __init__(self, *args, **kwargs):
        super().__init__(1, **kwargs)

    # fmt: off
    def __neg__      (self                             ) : return Scalar(-1)
    def __mul__      (self, o: NumericType | Expression) : return o
    def __matmul__   (self, o: NumericType | Expression) : return o
    def __pow__      (self, o: NumericType | Expression) : return self
    def __rmul__     (self, o: NumericType | Expression) : return o
    def __rmatmul__  (self, o: NumericType | Expression) : return o
    def __rtruediv__ (self, o: NumericType | Expression) : return o
    def __rpow__     (self, o: NumericType | Expression) : return o
    def __abs__      (self                             ) : return self
    # fmt: on


zero = ConstantZero()

one = ConstantOne()


class Piecewise(Expression):
    """PiecewiseFunction
    A piecewise function. 一维或多维，分段函数
    """

    def __init__(self, piecewise_func: typing.List[typing.Tuple[Expression | float | int, Expression]], **kwargs):
        super().__init__(None, **kwargs)
        self._piecewise = piecewise_func

    def __copy__(self) -> Piecewise:
        res = super().__copy__()
        res._piecewise = self._piecewise
        return res

    def __call__(self, *args, **kwargs) -> NumericType:
        if len(args) == 0:
            return self

        elif any([callable(val) for val in args]):
            return super().__call__(*args, **kwargs)

        elif isinstance(args[0], float):
            for func, cond in self._piecewise:
                if not cond(*args, **kwargs):
                    continue
                else:
                    res = func(*args, **kwargs)
                    break
            else:
                raise RuntimeError(f"Can not fit any condition! {args}")

            return res
        elif isinstance(args[0], array_type):
            res = np.full_like(args[0], np.nan)
            for func, cond in self._piecewise:
                marker = cond(*args, **kwargs)
                if callable(func):
                    _args = [(a[marker] if isinstance(a, array_type) else a) for a in args]
                    _kwargs = {k: (v[marker] if isinstance(v, array_type) else v) for k, v in kwargs.items()}
                    res[marker] = func(*_args, **_kwargs)
                else:
                    res[marker] = func

            return res
        else:
            raise TypeError(f"PiecewiseFunction only support single float or  1D array, {args}")


def piecewise(func_cond, size=None, **kwargs):
    if not isinstance(func_cond, list):
        raise TypeError(f"Illegal type {type(func_cond)}")
    elif all([isinstance(func, (array_type, float, int)) and isinstance(cond, array_type) for func, cond in func_cond]):
        res = np.full_like(func_cond[0][0], np.nan)
        for func, cond in func_cond:
            if np.sum(cond) == 0:
                continue
            if isinstance(func, array_type) and func.size == res.size:
                res[cond] = func[cond]
            else:
                res[cond] = func

        return res
    else:
        return Piecewise(func_cond, **kwargs)


def derivative(y, *args, order=1):
    if isinstance(y, array_type):
        from .Function import Function

        return Function(*args, y).d(*args)
    else:
        return Derivative(order, y, *args)


def integral(y, *args, order=1):
    if isinstance(y, array_type):
        from .Function import Function

        return Function(*args, y).I(*args)
    else:
        return Antiderivative(order, y, *args)


class Derivative(Expression):
    """算符: 用于表示一个运算符，可以是函数，也可以是类的成员函数
    受 np.ufunc 启发而来。
    可以通过 ExprOp(op, method=method) 的方式构建一个 ExprOp 对象。

    """

    def __init__(self, order, expr: Expression, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
        self._order = order
        self._expr = expr

    @property
    def order(self) -> int | None:
        return self._order

    @property
    def domain(self) -> DomainBase:
        return self._expr.domain

    def _render_latex_(self) -> str:
        if self._order == 0:
            return self._expr._render_latex_()
        elif self._order == 1:
            return f"d{self._expr._render_latex_()}"
        elif self._order > 1:
            return rf"d_{{\left[{self._order}\right]}}{self._expr._render_latex_()}"
        elif self._order < 0:
            return rf"\int_{{\left[{self._order}\right]}}{self._expr._render_latex_()}"

    def __eval__(self, *args, **kwargs):
        if self._op is None:
            if isinstance(self._expr, Variable):
                y = self._expr(*args)
                x = args[0]
            elif isinstance(self._expr, Expression):
                x = self._expr.domain.points[0]
                y = self._expr(x)

            self._op = interpolate(x, y).derivative(self._order)

        return super().__eval__(*args, **kwargs)


class Antiderivative(Derivative):
    def __init__(self, order, expr, **kwargs):
        super().__init__(-order, expr, **kwargs)

    def _render_latex_(self) -> str:
        if isinstance(self._order, (list, tuple)):
            return rf"\int_{{{self._order}}} \left({self._expr._render_latex_()} \right)"
        elif self._order == 1:
            return rf"\int \left({self._expr._render_latex_()} \right)"
        elif self._order == 2:
            return rf"\iint \left({self._expr._render_latex_()} \right)"
        else:
            return rf"\intop^{{{self._order}}}\left({self._expr._render_latex_()}\right)"

    def _eval(self, *args, **kwargs):
        ppoly, x = self._ppoly(*args, **kwargs)
        return ppoly.antiderivative(self._order)(x)


class PartialDerivative(Derivative):
    def __repr__(self) -> str:
        return f"d_{{{self._order}}} ({Expression._repr_s(self._expr)})"

    def __functor__(self):
        return self._expr.derivative(self._order) / self._expr


from ..numlib.smooth import smooth as _smooth

from scipy.signal import savgol_filter


class SmoothOp(Expression):
    def __init__(self, op, *args, **kwargs) -> None:
        super().__init__(op or savgol_filter, *args, options=kwargs)

    def __eval__(self, y: array_type, *args, **kwargs) -> typing.Any:
        if len(args) + len(kwargs) > 0:
            logger.warning(f"Ignore {args} {kwargs}")

        return self._op(y, **self._metadata.get("options", {}))

    def __call__(self, *args, **kwargs):
        if len(args) + len(kwargs) > 0:
            return super().__call__(*args, **kwargs)
        else:
            return self.__eval__(*self._children)


def smooth(expr, *args, op=None, **kwargs):
    if isinstance(expr, array_type):
        return SmoothOp(op, expr, *args, **kwargs)()
    else:
        return SmoothOp(op, expr, *args, **kwargs)
