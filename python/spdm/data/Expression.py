from __future__ import annotations

import typing
from copy import copy, deepcopy
import functools
import collections.abc
import numpy as np
import numpy.typing as np_tp
from .HTree import HTree, HTreeNode
from .Path import update_tree, Path
from .Functor import Functor
from .sp_property import SpTree
from ..utils.misc import group_dict_by_prefix
from ..utils.numeric import float_nan, meshgrid, bitwise_and
from ..utils.tags import _not_found_
from ..utils.typing import ArrayType, NumericType, array_type, as_array, is_scalar, is_array, numeric_type
from ..utils.logger import logger
from ..numlib.interpolate import interpolate


class DomainBase:
    """函数定义域"""

    _metadata = {"fill_value": float_nan}

    def __init__(self, *args, **kwargs) -> None:
        if len(args) > 0 and isinstance(args[0], dict):
            kwargs = update_tree(args[0], kwargs)
            args = args[1:]

        if len(args) == 0:
            pass
        elif "dims" in kwargs:
            raise RuntimeError(f"Redefine dims")
        elif len(args) > 1:
            kwargs["dims"] = args
        elif args[0] is not None:
            kwargs["dims"] = args[0]

        # if len(self.periods) > 0:
        #     dims = [as_array(v) for v in dims]
        #     periods = self.periods
        #     for idx in range(len(dims)):
        #         if periods[idx] is not np.nan and not np.isclose(dims[idx][-1] - dims[idx][0], periods[idx]):
        #             raise RuntimeError(
        #                 f"idx={idx} periods {periods[idx]} is not compatible with dims [{dims[idx][0]},{dims[idx][-1]}] "
        #             )
        #         if not np.all(dims[idx][1:] > dims[idx][:-1]):
        #             raise RuntimeError(
        #                 f"dims[{idx}] is not increasing! {dims[idx][:5]} {dims[idx][-1]} \n {dims[idx][1:] - dims[idx][:-1]}"
        #             )
        self._dims = kwargs.pop("dims", [])
        if len(kwargs) > 0:
            self._metadata = update_tree(deepcopy(self.__class__._metadata), kwargs)

    @property
    def is_simple(self) -> bool:
        return self._dims is not None

    @property
    def dims(self) -> typing.Tuple[ArrayType]:
        """函数的网格，即定义域的网格"""
        if self._dims is None or len(self._dims) == 0:
            raise RuntimeError(f"dims is not defined")
        return self._dims

    @property
    def ndims(self) -> int:
        return len(self._dims)

    @property
    def shape(self) -> typing.Tuple[int]:
        return tuple([len(d) for d in self.dims])

    @functools.cached_property
    def points(self) -> typing.Tuple[ArrayType]:
        if len(self.dims) == 1:
            return self.dims
        else:
            return meshgrid(*self.dims, indexing="ij")

    @functools.cached_property
    def bbox(self) -> typing.Tuple[typing.List[float], typing.List[float]]:
        """函数的定义域"""
        return tuple(([d[0], d[-1]] if not isinstance(d, float) else [d, d]) for d in self.dims)

    @property
    def periods(self):
        return None

    def mask(self, *args) -> bool | np_tp.NDArray[np.bool_]:
        # or self._metadata.get("extrapolate", 0) != 1:
        if self.dims is None or len(self.dims) == 0 or self._metadata.get("extrapolate", 0) != "raise":
            return True

        if len(args) != self.ndim:
            raise RuntimeError(f"len(args) != len(self.dims) {len(args)}!={len(self.dims)}")

        v = []
        for i, (xmin, xmax) in enumerate(self.bbox):
            v.append((args[i] >= xmin) & (args[i] <= xmax))

        return bitwise_and.reduce(v)

    def check(self, *x) -> bool | np_tp.NDArray[np.bool_]:
        """当坐标在定义域内时返回 True，否则返回 False"""

        d = [child.__check_domain__(*x) for child in self._children if hasattr(child, "__domain__")]

        if isinstance(self._func, Functor):
            d += [self._func.__domain__(*x)]

        d = [v for v in d if (v is not None and v is not True)]

        if len(d) > 0:
            return np.bitwise_and.reduce(d)
        else:
            return True

    def eval(self, func, *xargs, **kwargs):
        """根据 __domain__ 函数的返回值，对输入坐标进行筛选"""

        mask = self.__domain__().mask(*xargs)

        mask_size = mask.size if isinstance(mask, array_type) else 1
        masked_num = np.sum(mask)

        if not isinstance(mask, array_type) and not isinstance(mask, (bool, np.bool_)):
            raise RuntimeError(f"Illegal mask {mask} {type(mask)}")
        elif masked_num == 0:
            raise RuntimeError(f"Out of domain! {self} {xargs} ")

        if masked_num < mask_size:
            xargs = tuple(
                [
                    (
                        arg[mask]
                        if isinstance(mask, array_type) and isinstance(arg, array_type) and arg.ndim > 0
                        else arg
                    )
                    for arg in xargs
                ]
            )
        else:
            mask = None

        value = func._eval(*xargs, **kwargs)

        if masked_num < mask_size:
            res = value
        elif is_scalar(value):
            res = np.full_like(mask, value, dtype=self._type_hint())
        elif isinstance(value, array_type) and value.shape == mask.shape:
            res = value
        elif value is None:
            res = None
        else:
            res = np.full_like(mask, self.fill_value, dtype=self._type_hint())
            res[mask] = value
        return res


def guess_coords(holder, prefix="coordinate", **kwargs):
    if holder is None or holder is _not_found_:
        return None

    coords = []

    metadata = getattr(holder, "_metadata", {})

    dims_s, *_ = group_dict_by_prefix(metadata, prefix, sep=None)

    if dims_s is not None and len(dims_s) > 0:
        dims_s = {int(k): v for k, v in dims_s.items() if k.isdigit()}
        dims_s = dict(sorted(dims_s.items(), key=lambda x: x[0]))

        for c in dims_s.values():
            if not isinstance(c, str):
                d = as_array(c)
            elif c == "1...N":
                d = None
            # elif isinstance(holder, HTree):
            #     d = holder.get(c, _not_found_)
            else:
                d = Path(c).get(holder, _not_found_)

            if d is _not_found_ or d is None:
                # logger.warning(f"Can not get coordinates {c} from {holder}")
                coords = []
                break
            coords.append(as_array(d))

            # elif c.startswith("../"):
            #     d = as_array(holder._parent.get(c[3:], _not_found_))
            # elif c.startswith(".../"):
            #     d = as_array(holder._parent.get(c, _not_found_))
            # elif hasattr(holder.__class__, "get"):
            #     d = as_array(holder.get(c, _not_found_))
            # else:
            #     d = _not_found_
            # elif c.startswith("*/"):
            #     raise NotImplementedError(f"TODO:{self.__class__}.dims:*/")
            # else:
            #     d = as_array(holder.get(c, _not_found_))

    if len(coords) == 0:
        return guess_coords(getattr(holder, "_parent", None), prefix=prefix, **kwargs)
    else:
        return tuple(coords)


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

    Domain = DomainBase

    def __init__(self, expr: typing.Callable[..., NumericType], *children, domain=_not_found_, **kwargs) -> None:
        """
        Parameters

        args : typing.Any
            操作数
        op : typing.Callable  | ExprOp
            运算符，可以是函数，也可以是类的成员函数
        kwargs : typing.Any
            命名参数， 用于传递给运算符的参数

        """

        if expr is np.divide and is_scalar(children[1]) and children[1] == 0:
            logger.warning(f"Divide by zero: {expr} {children}")

        if self.__class__ is Expression and expr.__class__ is Expression and len(children) == 0:
            self.__copy_from__(expr)
            self._metadata = update_tree(self._metadata, kwargs)

        elif all([isinstance(v, np.ndarray) for v in [expr, *children]]):
            # 构建插值函数
            from .Function import Function

            self.__class__ = Function
            Function.__init__(self, expr, *children, domain=domain, **kwargs)
            return
        elif expr is None or callable(expr):
            self._op = expr
            self._children: typing.Tuple[typing.Type[Expression]] = children

        elif is_scalar(expr):
            match expr:
                case 0:
                    self.__class__ = ConstantZero
                    ConstantZero.__init__(self, domain=domain, **kwargs)
                case 1:
                    self.__class__ = ConstantOne
                    ConstantOne.__init__(self, domain=domain, **kwargs)
                case _:
                    self.__class__ = Scalar
                    Scalar.__init__(self, expr, domain=domain, **kwargs)
            return
        else:
            # dims = self.dims
            # value = self._value
            # if value is _not_found_ or value is None:
            #     self._func = None
            # elif isinstance(value, scalar_type):
            #     self._func = ConstantsFunc(value)
            # elif isinstance(value, array_type) and value.size == 1:
            #     value = np.squeeze(value).item()
            #     if not isinstance(value, scalar_type):
            #         raise RuntimeError(f"TODO:  {value}")
            #     self._func = ConstantsFunc(value)
            # elif all([(not isinstance(v, array_type) or v.size == 1) for v in dims]):
            #     self._func = DiracDeltaFun(value, [float(v) for v in self.dims])
            # elif all([(isinstance(v, array_type) and v.ndim == 1 and v.size > 0) for v in dims]):
            #     self._func = self._interpolate()
            # else:
            #     raise RuntimeError(f"TODO: {dims} {value}")

            # return self._func
            raise NotImplementedError(f"{expr} {children}")

        if isinstance(domain, collections.abc.Sequence) and len(domain) == 0:
            domain = _not_found_

        self._domain = domain

        super().__init__(None, **kwargs)

    def __copy__(self) -> Expression:
        """复制一个新的 Expression 对象"""
        other: Expression = object.__new__(self.__class__)
        other.__copy_from__(self)
        return other

    def __copy_from__(self, other: Expression) -> Expression:
        """复制 other 到 self"""
        if isinstance(other, Expression):
            self._op = copy(other._op)
            self._children = copy(other._children)
            self._domain = copy(other._domain)
            self._metadata = copy(other._metadata)
        else:
            raise TypeError(f"{type(other)}")
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
        if method != "__call__" or len(kwargs) > 0:
            return Expression(Functor(ufunc, method=method, **kwargs), *args)
        else:
            return Expression(ufunc, *args)

    def __array__(self) -> ArrayType:
        """在定义域上计算表达式。"""
        if not is_array(self._cache):
            self._cache = self.__call__(*self.domain.points)
            if not isinstance(self._cache, (np.ndarray, float, int, bool)):
                raise RuntimeError(f"Can not calcuate! {self._cache}")

        return self._cache

    @property
    def domain(self) -> Domain:
        """返回表达式的定义域"""
        if not isinstance(self._domain, DomainBase):
            if self._domain is None or self._domain is _not_found_:
                self._domain = guess_coords(self)

            if self._domain is not None:
                self._domain = self.__class__.Domain(self._domain)

        if self._domain is None:
            for child in self._children:
                d = getattr(child, "domain", _not_found_)
                if d is not _not_found_ and d is not None:
                    self._domain = d
                    break
        # if self._domain is None:
        #     raise RuntimeError(f"Can not get domain! {self} ")
        return self._domain

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
    def name(self) -> str:
        return self._metadata.get("name", f"<{self.__class__.__name__}>")

    @property
    def __label__(self) -> str:
        return self._metadata.get("label", None) or self.name

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} label='{self.__label__}' />"

    def __repr__(self) -> str:
        return self.__label__

    @staticmethod
    def _repr_s(expr: Expression) -> str:
        if isinstance(expr, (bool, int, float, complex)):
            res = f"{{{expr}}}"
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
            res = expr._repr_latex_().strip("$")
        else:
            res = f"{{{expr}}}"
        return res

    def _repr_latex_(self) -> str:
        """for jupyter notebook display"""

        # label = self._metadata.get("label", None) or self._metadata.get("name", None)
        # if label is not None:
        #     return label

        nin = len(self._children)

        # op = self._metadata.get("label", None) or self._metadata.get("name", None)

        if isinstance(self._op, Expression):
            op = self._op.__label__
        elif isinstance(self._op, np.ufunc):
            op = EXPR_OP_TAG.get(self._op.__name__, self._op.__name__)
            nin = self._op.nin
        else:
            op = self._op.__class__.__name__

        children = [Expression._repr_s(child) for child in self._children]

        match nin:
            case 0:
                res = f"{op}"

            case 1:
                if op == "-":
                    res = f"- {children[0]}"

                elif not op.startswith("\\"):
                    res = rf"{op}{children[0]}"

                else:
                    res = rf"{op}{{{children[0]}}}"

            case 2:
                match op:
                    case "/":
                        res = f"\\frac{{{children[0]}}}{{{children[1]}}}"
                    case _:
                        res = rf"{children[0]} {op} {children[1]}"

            case _:
                res = rf"{op}{','.join(children)}"

        return rf"$$\left({res}\right)$$"

    @property
    def dtype(self):
        return self._type_hint()

    def _type_hint(self, *args):
        """TODO:获取表达式的类型"""
        return float

    def __functor__(self) -> Functor:
        """获取表达式的运算符，若为 constants 函数则返回函数值"""
        return self._op

    def _eval(self, *args, **kwargs):
        func = self.__functor__()

        if func is None:
            value = np.nan
        elif callable(func):
            try:
                value = func(*args, **kwargs)
            except Exception as error:
                raise RuntimeError(f"Failure to calculate  equation {self._repr_latex_()} !") from error

        elif isinstance(func, numeric_type):
            value = func

        else:
            raise RuntimeError(f"Unknown functor {func} {type(func)}")

        return value

    def __call__(self, *args, **kwargs) -> typing.Any:
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
            return self
        elif any([(isinstance(arg, Expression) or callable(arg)) for arg in args]):
            return Expression(self, *args, **kwargs, **self._metadata)

        if len(self._children) > 0:  # Traverse children
            children = []
            for child in self._children:
                if callable(child):
                    value = child(*args, **kwargs)
                elif hasattr(child, "__value__"):
                    value = child.__value__
                elif hasattr(child, "__array__"):
                    value = child.__array__()
                else:
                    value = child
                children.append(value)

            args = tuple(children)
            kwargs = {}

        if getattr(self, "CHECK_DOMAIN", False):
            res = self.domain.eval(self, *args, **kwargs)
        else:
            res = self._eval(*args, **kwargs)
        return res

    def integral(self, **kwargs) -> float:
        raise NotImplementedError(f"TODO:integral")

    def derivative(self, d, *args, **kwargs) -> typing.Type[Derivative]:
        if d == 0:
            return self
        elif isinstance(d, int) and d < 0:
            return Antiderivative(-d, *args, self, domain=self.domain, _parent=self._parent, **kwargs)
        else:
            return Derivative(d, *args, self, domain=self.domain, _parent=self._parent, **kwargs)

    def pd(self, *d) -> Expression:
        return self.derivative(d)

    @property
    def d(self) -> Expression:
        """1st derivative 一阶导数"""
        expr = self.derivative(1)
        expr._metadata["label"] = rf"$d\left({self.__label__}\right)$"
        return expr

    @property
    def d2(self) -> Expression:
        """2nd derivative 二阶导数"""
        return self.derivative(2)

    @property
    def I(self) -> Expression:
        """antiderivative 原函数"""
        return self.derivative(-1)

    @property
    def dln(self) -> Expression:
        """logarithmic derivative 对数求导"""
        expr = self.d / self
        expr._metadata["label"] = rf"$dln\left({self.__label__}\right)$"
        return expr

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
        if isinstance(self._idx, str):
            return kwargs.get(self._idx, None)
        elif self._idx < len(args):
            return args[self._idx]
        else:
            raise RuntimeError(f"Variable {self.__label__} require {self._idx+1} args, but only {len(args)} provided!")
        # return args[self._idx]

    def __repr__(self) -> str:
        return self.__label__


class Scalar(Expression):
    def __init__(self, value, *args, **kwargs) -> None:
        super().__init__(None, *args, **kwargs)
        self._value = value

    @property
    def __label__(self):
        return self._value

    def __str__(self):
        return str(self._value)

    def __repr__(self) -> str:
        return str(self._value)

    def __equal__(self, other) -> bool:
        return self._value == other

    def __call__(self, *args, **kwargs):
        return self._value

    def derivative(self, *args, **kwargs):
        return ConstantZero(domain=self.domain, _parent=self._parent, **kwargs)


class ConstantZero(Scalar):
    def __init__(self, *args, **kwargs):
        super().__init__(0, **kwargs)


class ConstantOne(Scalar):
    def __init__(self, *args, **kwargs):
        super().__init__(1, **kwargs)


zero = ConstantZero()
one = ConstantOne()


class Derivative(Expression):
    """算符: 用于表示一个运算符，可以是函数，也可以是类的成员函数
    受 np.ufunc 启发而来。
    可以通过 ExprOp(op, method=method) 的方式构建一个 ExprOp 对象。

    """

    def __init__(self, order, expr, **kwargs):
        super().__init__(None, **kwargs)
        self._expr = expr
        self._order = order

    @property
    def order(self) -> int | None:
        return self._order

    def __repr__(self) -> str:
        return f"d{Expression._repr_s(self._expr)}"

    def _repr_latex_(self) -> str:
        return f"$d{Expression._repr_s(self._expr)}$"

        # return rf"\frac{{d({Expression._repr_s(self._children[1])})}}{{{Expression._repr_s(self._children[0])}}}"

    def _ppoly(self, *args, **kwargs):
        if isinstance(self._expr, Variable):
            y = self._expr(*args)
            x = args[0]
        elif isinstance(self._expr, Expression):
            x = self._expr.domain.points[0]
            y = self._expr.__array__()
        return interpolate(x, y), args[0]

    def _eval(self, *args, **kwargs):
        ppoly, x = self._ppoly(*args, **kwargs)
        return ppoly.derivative(self._order)(x)


class LogDerivative(Derivative):
    def __repr__(self) -> str:
        return f"d \\ln {Expression._repr_s(self._expr)}"

    def __functor__(self):
        return self._expr.derivative(self._order) / self._expr


class PartialDerivative(Derivative):
    def __repr__(self) -> str:
        return f"d_{{{self._order}}} ({Expression._repr_s(self._expr)})"

    def __functor__(self):
        return self._expr.derivative(self._order) / self._expr


class Antiderivative(Derivative):
    def __repr__(self) -> str:
        if isinstance(self._order, (list, tuple)):
            return rf"\int_{{{self._order}}} \left({self._expr.__repr__()} \right)"
        elif self._order == 1:
            return rf"\int \left({self._expr.__repr__()} \right)"
        elif self._order == 2:
            return rf"\iint \left({self._expr.__repr__()} \right)"
        else:
            return rf"\intop^{{{self._order}}}\left({self._expr.__repr__()}\right)"

    def _eval(self, *args, **kwargs):
        ppoly, x = self._ppoly(*args, **kwargs)
        return ppoly.antiderivative(self._order)(x)


class Piecewise(Expression):
    """PiecewiseFunction
    A piecewise function. 一维或多维，分段函数
    """

    def __init__(self, func: typing.List[Expression | float | int], cond: typing.List[typing.Callable], **kwargs):
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
            return super().__call__(x, *args, **kwargs)

            # raise TypeError(f"PiecewiseFunction only support single float or  1D array, {type(x)} {array_type}")
