from __future__ import annotations

import typing
import functools
from typing_extensions import Self
import collections
import numpy as np
import numpy.typing as np_tp
from copy import copy, deepcopy

from ..numlib.interpolate import interpolate
from ..utils.logger import logger
from ..utils.tags import _not_found_
from ..utils.typing import ArrayType, NumericType, array_type, get_args, get_origin, as_array
from ..utils.misc import group_dict_by_prefix
from .Expression import Expression, zero
from .Functor import Functor
from .Path import update_tree, Path
from .Domain import DomainBase
from .HTree import List


def guess_coords(holder, prefix="coordinate", **kwargs):
    if holder is None or holder is _not_found_:
        return None
    elif isinstance(holder, List):
        return guess_coords(holder._parent, prefix=prefix, **kwargs)

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


class Function(Expression):
    """
    Function

    A function is a mapping between two sets, the _domain_ and the  _value_.
    The _value_  is the set of all possible outputs of the function.
    The _domain_ is the set of all possible inputs  to the function.

    函数定义域为多维空间时，网格采用rectlinear mesh，即每个维度网格表示为一个数组 _dims_ 。
    """

    Domain = DomainBase

    def __init__(self, *xy: array_type, **kwargs):
        """
        Parameters
        ----------
        *x : typing.Tuple[ArrayType]
            自变量
        y : ArrayType
            变量
        kwargs : 命名参数，
                *           : 用于传递给 Node 的参数
        extrapolate: int |str
            控制当自变量超出定义域后的值
            * if ext=0  or 'extrapolate', return the extrapolated value. 等于 定义域无限
            * if ext=1  or 'nan', return nan
        """
        super().__init__(None, **kwargs)

        if any([x is _not_found_ for x in xy]):
            raise ValueError(f"Illegal arguments {xy}")

        if len(xy) == 1 and isinstance(xy[0], tuple):
            xy = xy[0]

        match len(xy):
            case 0:
                raise RuntimeError(f"illegal x,y {xy} ")

            case 1:
                x = []
                y = xy[-1]

            case _:
                x = xy[:-1]
                y = xy[-1]

        if len(x) == 0:
            x = guess_coords(self)

        if x is None or len(x) == 0:
            x = [np.linspace(0, 1, y.size)]

        self._dims = tuple(x)

        self._cache = y

    def __copy__(self) -> Self:
        """copy from other"""
        other: Function = super().__copy__()
        other._dims = self._dims
        other._cache = self._cache
        return other

    def __repr__(self) -> str:
        return f"{self.__label__}"

    def __getitem__(self, idx) -> NumericType:
        return self._cache[idx]

    def __setitem__(self, idx, value) -> None:
        self._op = None
        self._cache[idx] = value

    @property
    def x_label(self) -> str:
        return self._metadata.get("x_label", "[-]")

    @functools.cached_property
    def domain(self) -> DomainBase:
        return self.__class__.Domain(*self.dims)

    @property
    def dims(self) -> typing.Tuple[array_type, ...]:
        """函数的网格，即定义域的网格"""
        return self._dims

    @property
    def ndim(self) -> int:
        """函数的维度，函数所能接收参数的个数。"""
        return len(self._dims)

    @property
    def rank(self) -> int:
        """函数的秩，rank=1 标量函数， rank=3 矢量函数 None 待定"""
        return self._cache.shape[self.ndim :] if len(self._cache.shape) > self.ndim else 1

    @property
    def _ppoly(self):
        if self._op is None:
            if not isinstance(self._cache, array_type):
                raise RuntimeError(f"self._cache is not array_type! {(self._cache)}")

            if len(self._dims) == 0:
                raise RuntimeError(f" Dimension is not defined!   {self._dims} ")

            periods = (self._metadata.get("periods", None),)
            extrapolate = (self._metadata.get("extrapolate", 0),)

            self._op = interpolate(*self._dims, self._cache, periods=periods, extrapolate=extrapolate)
        return self._op

    def __eval__(self, *args, **kwargs):
        """
        对函数进行编译，用插值函数替代原始表达式，提高运算速度

        NOTE：
            - 由 points，value  生成插值函数，并赋值给 self._ppoly。 插值函数相对原始表达式的优势是速度快，缺点是精度低。
            - 当函数为expression时，调用 value = self.__call__(*points) 。
        TODO:
            - 支持 JIT 编译, support JIT compile
            - 优化缓存
            - 支持多维插值
            - 支持多维求导，自动微分 auto diff

        Parameters
        ----------
        d : typing.Any
            order of function
        force : bool
            if force 强制返回多项式ppoly ，否则 可能返回 Expression or callable
        """

        res = self._ppoly(*args, **kwargs)

        return res

    def derivative(self, order, *args, **kwargs) -> Expression:
        if len(self.__array__().shape) == 0:
            return zero
        else:
            return Expression(
                self._ppoly.derivative(order, *args, **kwargs),
                _parent=self._parent,
                **collections.ChainMap({"label": rf"d_{{{order}}} {self.__repr__()}"}, self._metadata),
            )

    def antiderivative(self, order: int, *args, **kwargs) -> Expression:
        if len(self.__array__().shape) == 0:
            return zero
        else:
            label = rf"\int_{{{order}}} {self.__repr__()}" if order > 1 else rf"\int {self.__repr__()}"
            return Expression(
                self._ppoly.derivative(-order, *args, **kwargs),
                _parent=self._parent,
                **collections.ChainMap({"label": label}, self._metadata),
            )

    def partial_derivative(self, order: typing.Tuple[int, ...], *args, **kwargs) -> Expression:
        if len(self.__array__().shape) == 0:
            return zero
        else:
            return Expression(
                self._ppoly.derivative(order, *args, **kwargs),
                _parent=self._parent,
                **collections.ChainMap({"label": rf"d_{{{order}}} {self.__repr__()}"}, self._metadata),
            )

    def integral(self, *args, **kwargs) -> float:
        raise NotImplementedError(f"TODO:integral")

    def validate(self, value=None, strict=False) -> bool:
        """检查函数的定义域和值是否匹配"""

        m_shape = tuple(self.shape)

        v_shape = ()

        if value is None:
            value = self._cache

        if value is None:
            raise RuntimeError(f" value is None! {self.__str__()}")

        if isinstance(value, array_type):
            v_shape = tuple(value.shape)

        if (v_shape == m_shape) or (v_shape[:-1] == m_shape):
            return True
        elif strict:
            raise RuntimeError(f" value.shape is not match with dims! {v_shape}!={m_shape} ")
        else:
            logger.warning(f" value.shape is not match with dims! {v_shape}!={m_shape} ")
            return False


class Polynomials(Expression):
    """A wrapper for numpy.polynomial
    TODO: imcomplete
    """

    def __init__(
        self,
        coeff,
        *args,
        type: str = None,
        domain=None,
        window=None,
        symbol="x",
        preprocess=None,
        postprocess=None,
        **kwargs,
    ) -> None:
        match type:
            case "chebyshev":
                from numpy.polynomial.chebyshev import Chebyshev

                Op = Chebyshev
            case "hermite":
                from numpy.polynomial.hermite import Hermite

                Op = Hermite
            case "hermite":
                from numpy.polynomial.hermite_e import HermiteE

                Op = HermiteE
            case "laguerre":
                from numpy.polynomial.laguerre import Laguerre

                Op = Laguerre
            case "legendre":
                from numpy.polynomial.legendre import Legendre

                Op = Legendre
            case _:  # "power"
                import numpy.polynomial.polynomial as polynomial

                Op = polynomial

        op = Op(coeff, domain=domain, window=window, symbol=symbol)

        super().__init__(op, *args, **kwargs)
        self._preprocess = preprocess
        self._postprocess = postprocess

    def __eval__(self, x: array_type | float, *args, **kwargs) -> array_type | float:
        if len(args) + len(kwargs) > 0:
            logger.warning(f"Ignore arguments {args} {kwargs}")

        if not isinstance(x, (array_type, float)):
            return super().__call__(x)
        if self._preprocess is not None:
            x = self._preprocess(x)

        y = self._op(x)

        if self._postprocess is not None:
            y = self._postprocess(y)

        return y


def function_like(y: NumericType, *args: NumericType, **kwargs) -> Function:
    if len(args) == 0 and isinstance(y, Function):
        return y
    else:
        return Function(y, *args, **kwargs)
