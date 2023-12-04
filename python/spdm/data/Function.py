from __future__ import annotations

import typing
import collections
import numpy as np

from ..numlib.interpolate import interpolate
from ..utils.logger import logger
from ..utils.tags import _not_found_
from ..utils.typing import ArrayType, NumericType, array_type, get_args, get_origin, as_array
from .Expression import Expression
from .Functor import Functor


class Function(Expression):
    """
    Function

    A function is a mapping between two sets, the _domain_ and the  _value_.
    The _value_  is the set of all possible outputs of the function.
    The _domain_ is the set of all possible inputs  to the function.

    函数定义域为多维空间时，网格采用rectlinear mesh，即每个维度网格表示为一个数组 _dims_ 。
    """

    def __init__(self, *xy, domain=None, **kwargs):
        """
        Parameters
        ----------
        y : ArrayType
            变量
        x : typing.Tuple[ArrayType]
            自变量

        kwargs : 命名参数，
                *           : 用于传递给 Node 的参数
        extrapolate: int |str
            控制当自变量超出定义域后的值
            * if ext=0  or 'extrapolate', return the extrapolated value. 等于 定义域无限
            * if ext=1  or 'nan', return nan
        """
        if len(xy) == 0:
            raise RuntimeError(f"illegal x,y {xy} ")
        elif len(xy) == 1 and isinstance(xy[0], tuple):
            xy = xy[0]

        # elif len(xy) == 1 and isinstance(xy[0], tuple):
        #     xy = xy[0]

        if callable(xy[-1]):
            func = xy[-1]
            value = _not_found_
        else:
            func = None
            value = as_array(xy[-1])

        if len(xy) <= 1:
            pass
        else:
            if domain is None or domain is _not_found_:
                domain = {}
            if isinstance(domain, dict):
                domain["dims"] = xy[:-1]
            # else:
            #     raise RuntimeError(f"illegal domain={domain}")
            #
        super().__init__(func, domain=domain, **kwargs)
        self._cache = value

    def __copy_from__(self, other: Function) -> Function:
        """copy from other"""
        Expression.__copy_from__(self, other)
        if isinstance(other, Function):
            self._dims = other._dims
            self._cache = other._cache
            return self

    def __repr__(self) -> str:
        return f"{self.__label__}"

    def __getitem__(self, idx) -> NumericType:
        return self._cache[idx]

    def __setitem__(self, idx, value) -> None:
        self._cache[idx] = value

    @property
    def x_label(self) -> str:
        return self._metadata.get("x_label", "[-]")

    @property
    def dims(self) -> typing.Tuple[ArrayType]:
        """函数的网格，即定义域的网格"""
        return self.domain.dims

    @property
    def ndim(self) -> int:
        """函数的维度，函数所能接收参数的个数。"""
        return self.domain.ndims

    @property
    def rank(self) -> int:
        """函数的秩，rank=1 标量函数， rank=3 矢量函数 None 待定"""
        return 1

    def _type_hint(self) -> typing.Type:
        tp = get_args(get_origin(self))
        if len(tp) == 0:
            return float
        else:
            return tp[-1]

    def __functor__(self) -> Functor:
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
        if self._op is None:
            self._op = self._interpolate()
        return self._op

    def _interpolate(self):
        if not isinstance(self._cache, array_type):
            raise RuntimeError(f"self._cache is not array_type! {(self._cache)}")

        if self.domain is None:
            raise RuntimeError(f"{self}")

        return interpolate(
            *self.domain.dims,
            self._cache,
            periods=self.domain.periods,
            extrapolate=self._metadata.get("extrapolate", 0),
        )

    def __call__(self, *args, **kwargs) -> typing.Any:
        if len(args) == 0 and len(kwargs) == 0:
            return self
        else:
            return super().__call__(*args, **kwargs)

    def derivative(self, *d, **kwargs) -> Function:
        if len(self.__array__().shape) == 0:
            return Function(*self.dims, 0.0)

        if len(d) == 0:
            d = [1]

        if len(d) > 1:
            return Function(
                self._interpolate().partial_derivative(*d, **kwargs),
                domain=self.domain,
                _parent=self._parent,
                **collections.ChainMap({"label": rf"d_{{[{d}]}} {self.__repr__()}"}, self._metadata),
            )
        elif d[0] < 0:
            if len(d) > 1:
                logger.warning(f"ignore {d[1:]} ")
            func = self._interpolate().antiderivative(-d[0], **kwargs)
            return Function(
                func,
                domain=self.domain,
                _parent=self._parent,
                **collections.ChainMap({"label": rf"\int {self.__repr__()}"}, self._metadata),
            )
        else:
            return Function(
                self._interpolate().derivative(*d, **kwargs),
                domain=self.domain,
                _parent=self._parent,
                **collections.ChainMap({"label": rf"d({self.__repr__()})"}, self._metadata),
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


def function_like(y: NumericType, *args: NumericType, **kwargs) -> Function:
    if len(args) == 0 and isinstance(y, Function):
        return y
    else:
        return Function(y, *args, **kwargs)
