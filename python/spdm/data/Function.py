from __future__ import annotations

import typing

import numpy as np

from ..numlib.interpolate import interpolate
from ..utils.logger import logger
from ..utils.typing import ArrayType, NumericType, array_type, get_args, get_origin
from .Expression import Expression
from .Functor import Functor


class Function(Expression):
    """
    Function
    ---------
    A function is a mapping between two sets, the _domain_ and the  _value_.
    The _value_  is the set of all possible outputs of the function.
    The _domain_ is the set of all possible inputs  to the function.

    函数定义域为多维空间时，网格采用rectlinear mesh，即每个维度网格表示为一个数组 _dims_ 。
    """

    def __init__(self, *xy, **kwargs):
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
        if len(xy) == 0 or not all([isinstance(v, np.ndarray) for v in xy]):
            raise RuntimeError(f"illegal x,y {xy} ")

        self._value = xy[-1]

        Expression.__init__(self, None, domain=xy[:-1], **kwargs)

    def __copy_from__(self, other: Function) -> Function:
        """copy from other"""
        Expression.__copy_from__(self, other)
        if isinstance(other, Function):
            self._dims = other._dims
            self._value = other._value
            return self

    def __repr__(self) -> str:
        return f"{self.__label__}"

    def _repr_svg_(self) -> str:
        try:
            from ..view import View as sp_view

            res = sp_view.plot(
                ((self.dims[0], self.__array__()), {"label": self.__label__, "x_label": self.x_label}), output="svg"
            )
        except Exception as error:
            # logger.error(error)
            res = None
        return res

    def __getitem__(self, idx) -> NumericType:
        return self._value[idx]

    def __setitem__(self, idx, value) -> None:
        self._value[idx] = value

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
        if self._func is None:
            self._func = self._interpolate()
        return self._func

    def __array__(self, *args, **kwargs) -> NumericType:
        return self._value

    def _interpolate(self):
        if not isinstance(self._value, array_type):
            raise RuntimeError(f"self.__array__ is not array_type! {(self._value)}")
        return interpolate(
            *self.domain.dims,
            self._value,
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
            return Function(0.0, self.dims)

        if len(d) == 0:
            d = [1]

        if len(d) > 1:
            return Function(
                self._interpolate().partial_derivative(*d, **kwargs),
                *self.dims,
                periods=self.periods,
                **self._metadata,
            )
        elif d[0] < 0:
            return Function(
                self._interpolate().antiderivative(*d, **kwargs),
                *self.dims,
                periods=self.periods,
                label=rf"\int {self.__repr__()}",
            )
        else:
            return Function(
                self._interpolate().derivative(*d, **kwargs),
                *self.dims,
                periods=self.periods,
                **self._metadata,
            )

    def integral(self, *args, **kwargs) -> float:
        raise NotImplementedError(f"TODO:integral")

    def validate(self, value=None, strict=False) -> bool:
        """检查函数的定义域和值是否匹配"""

        m_shape = tuple(self.shape)

        v_shape = ()

        if value is None:
            value = self._value

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
