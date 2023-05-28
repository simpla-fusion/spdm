from __future__ import annotations

import collections.abc
import inspect
import typing
from copy import copy
from enum import Enum
import functools
import numpy as np
from scipy.interpolate import (InterpolatedUnivariateSpline,
                               RectBivariateSpline, RegularGridInterpolator,
                               UnivariateSpline, interp1d, interp2d)
from spdm.data.Expression import Expression
from spdm.utils.typing import ArrayType, NumericType
import numpy.typing
from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix, try_get
from ..utils.tags import _not_found_
from ..utils.typing import (ArrayType, NumericType, array_type, numeric_type,
                            scalar_type)
from .Expression import Expression

_T = typing.TypeVar("_T")


class Function(Expression[_T]):
    """
        Function
        ---------
        A function is a mapping between two sets, the _domain_ and the  _value_.
        The _value_  is the set of all possible outputs of the function.
        The _domain_ is the set of all possible inputs  to the function.

        函数定义域为多维空间时，网格采用rectlinear mesh，即每个维度网格表示为一个数组 _dims_ 。

    """

    def __init__(self, value: NumericType | Expression, *dims: ArrayType, periods=None, **kwargs):
        """
            Parameters
            ----------
            value : NumericType
                函数的值
            mesh : typing.List[ArrayType]
                函数的定义域
            args : typing.Any
                位置参数, 用于与mesh_*，coordinate* 一起构建 mesh
            kwargs : typing.Any
                命名参数，
                    *           : 用于传递给 Node 的参数

        """

        if isinstance(value, Expression) or callable(value) or (isinstance(value, tuple) and callable(value[0])):
            op = value
            value = None
        else:
            op = None

        if value is not None:
            try:
                value = np.asarray(value)
            except Exception as e:
                raise ValueError(f"Function.__init__  value should has numeric type, not {type(value)}! ") from e

        super().__init__(op, **kwargs)

        self._value = value

        self._dims = [np.asarray(v) for v in dims] if len(dims) > 0 else None

        self._periods = periods

        self._ppoly = None

        # if any(len(d.shape) > 1 for d in self.dims):
        #     raise RuntimeError(f"Function.__init__ incorrect dims {dims}! {self.__str__()}")

    def validate(self, value=None, strict=False) -> bool:
        """ 检查函数的定义域和值是否匹配 """

        m_shape = tuple(self.shape)

        v_shape = ()

        if value is None:
            value = self.__value__()

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

    def __duplicate__(self) -> Function:
        """ 复制一个新的 Function 对象 """
        other: Function = super().__duplicate__()
        other._value = self._value
        other._dims = self._dims
        other._periods = self._periods
        return other

    def __serialize__(self) -> typing.Mapping: raise NotImplementedError(f"")

    @classmethod
    def __deserialize__(cls, desc: typing.Mapping) -> Function: raise NotImplementedError(f"")

    @property
    def empty(self) -> bool: return self._value is None and self.dims is None and super().empty

    @property
    def dimensions(self) -> typing.List[ArrayType]: return self.dims
    """ for rectlinear mesh 每个维度对应一个一维数组，为网格的节点。"""

    @property
    def dims(self) -> typing.List[ArrayType]: return self._dims
    """ alias of dimensions """

    @property
    def ndim(self) -> int: return len(self.dims) if self.dims is not None else 0
    """ 函数的维度，即定义域的秩 """

    @property
    def shape(self) -> typing.Tuple[int]: return tuple(len(d) for d in self.dims) if self.dims is not None else tuple()
    """ 所需数组的形状 """

    @property
    def periods(self) -> typing.Tuple[float]: return self._periods

    @functools.cached_property
    def points(self) -> typing.List[ArrayType]:
        if self.dims is None:
            raise RuntimeError(self.dims)
        elif len(self.dims) == 1:
            logger.debug(self.dims)
            return self.dims[0]
        else:
            return np.meshgrid(*self.dims, indexing="ij")

    def __domain__(self, *args) -> bool:

        if self.dims is None or len(self.dims) == 0:
            return True

        if len(args) != len(self.dims):
            raise RuntimeError(f"len(args) != len(self.dims) {len(args)}!={len(self.dims)}")

        v = []
        for i, d in enumerate(self.dims):
            if not isinstance(d, array_type):
                v.append(np.isclose(args[i], d))
            elif len(d.shape) == 0:
                v.append(np.isclose(args[i], d.item()))
            elif len(d) == 1:
                v.append(np.isclose(args[i], d[0]))
            else:
                v.append((args[i] >= d[0]) & (args[i] <= d[-1]))
        return np.bitwise_and.reduce(v)

    # @property
    # def rank(self) -> int:
    #     """ 函数的秩，rank=1 标量函数， rank=3 矢量函数 None 待定 """
    #     if isinstance(self._value, array_type):
    #         return self._value.shape[-1]
    #     elif isinstance(self._value, tuple):
    #         return len(self._value)
    #     else:
    #         logger.warning(f"Function.rank is not defined!  {type(self._value)} default=1")
    #         return 1

    def __value__(self) -> ArrayType:
        """ 返回函数的数组 self._value """
        value = self._value

        if isinstance(value, Expression) or callable(value):
            value = value(*self.points)

        if not isinstance(value, array_type) and not value:
            value = None

        if value is None:
            pass
        elif isinstance(value, array_type):
            try:
                value = np.asarray(value)
            except Exception as error:
                raise TypeError(f"{type(value)} {value}") from error

            if not isinstance(value, array_type):
                raise RuntimeError(f"Function.compile() incorrect value {self.__str__()} value={value} ")
            elif value.size == 1:
                value = np.squeeze(value).item()

            if not np.isscalar(value) and not isinstance(value, array_type):
                raise TypeError(f"{type(value)} {value}")

        self._value = value

        return value

    def __array__(self, dtype=None, *args,  **kwargs) -> ArrayType:
        """ 重载 numpy 的 __array__ 运算符
                若 self._value 为 array_type 或标量类型 则返回函数执行的结果
        """
        res = self.__value__()

        if res is None or res is _not_found_ and self.callable:
            res = self._value = self._eval(*self.points)

        if isinstance(res, numeric_type):
            res = np.asarray(res, dtype=self.__type_hint__ if dtype is None else dtype)
        else:
            raise TypeError(f" Can not get value {(res)}! fun={self.__str__()}")
        return res

    def __getitem__(self, idx) -> NumericType: return self.__value__()[idx]
    # raise NotImplementedError(f"Function.__getitem__ is not implemented!")

    def __setitem__(self, *args) -> None: raise RuntimeError("Function.__setitem__ is prohibited!")

    @property
    def __op__(self) -> typing.Callable:
        if self._op is None or self._op is _not_found_:
            self._op = self._compile()
        return self._op
        # if self.ndim == 1 and len(d) > 0:
        #     if len(d) != 1:
        #         raise RuntimeError(f" Univariate function has not partial_derivative!")
        #     ppoly = self._compile(**kwargs)
        #     if isinstance(ppoly, tuple):
        #         ppoly, opts, *_ = ppoly
        #     else:
        #         opts = None

        #     ppoly = ppoly.derivative(d[0])
        #     if isinstance(opts, collections.abc.Mapping):
        #         return ppoly, opts
        #     else:
        #         return ppoly
        # elif self.ndim > 1 and len(d) > 0:
        #     ppoly = self._compile(**kwargs)
        #     if isinstance(ppoly, tuple):
        #         ppoly, opts, *_ = ppoly
        #     else:
        #         opts = None
        #     if all(v >= 0 for v in d):
        #         ppoly = ppoly.partial_derivative(*d)
        #     elif all(v <= 0 for v in d):
        #         ppoly = ppoly.antiderivative(*[-v for v in d])
        #     else:
        #         raise RuntimeError(f"illegal derivative order {d}")

        #     if isinstance(opts, collections.abc.Mapping):
        #         return ppoly, opts
        #     else:
        #         return ppoly

    def _compile(self, *args, check_nan=True, force=False, **kwargs) -> typing.Callable[..., NumericType] | NumericType:
        """ 对函数进行编译，用插值函数替代原始表达式，提高运算速度

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

        if self._ppoly is not None and not force:
            logger.debug(self._ppoly)
            return self._ppoly

        value = self.__value__()

        if callable(value):
            return value
        elif np.isscalar(value):
            return value
        elif isinstance(value, array_type) and len(value.shape) == 0:
            return value.item()

        if not isinstance(value, array_type):
            return None

        m_shape = self.shape

        if len(value.shape) > len(m_shape):
            raise NotImplementedError(
                f"TODO: interpolate for rank >1 . {value.shape}!={m_shape}!  func={self.__str__()} ")
        elif tuple(value.shape) != tuple(m_shape):
            raise RuntimeError(
                f"Function.compile() incorrect value shape {value.shape}!={m_shape}! value={value} func={self.__str__()} ")

        if len(self.dims) == 1:
            x = self.dims[0]
            if check_nan:
                mark = np.isnan(value)
                nan_count = np.count_nonzero(mark)
                if nan_count > 0:
                    logger.warning(
                        f"{self.__class__.__name__}[{self.__str__()}]: Ignore {nan_count} NaN at {np.argwhere(mark)}.")
                    value = value[~mark]
                    x = x[~mark]

            ppoly = InterpolatedUnivariateSpline(x, value)
        elif len(self.dims) == 2:
            if check_nan:
                mark = np.isnan(value)
                nan_count = np.count_nonzero(mark)
                if nan_count > 0:
                    logger.warning(
                        f"{self.__class__.__name__}[{self.__str__()}]: Replace  {nan_count} NaN by 0 at {np.argwhere(mark)}.")
                    value[mark] = 0.0

            if isinstance(self.periods, collections.abc.Sequence):
                logger.warning(f"TODO: periods={self.periods}")

            ppoly = RectBivariateSpline(*self.dims, value), {"grid": False}
        else:
            raise NotImplementedError(f"Multidimensional interpolation for n>2 is not supported.! ndim={self.ndim} ")

        self._ppoly = ppoly
        return ppoly

    def compile(self, *args, **kwargs) -> Function:
        op, *opts = self._compile(*args, **kwargs)
        if len(opts) == 0:
            pass
        elif len(opts) > 0:
            opts = opts[0]
            op = functools.partial(op, **opts)
            if len(opts) > 1:
                logger.warning(f"Function.compile() ignore opts! {opts[1:]}")
        if op is None:
            raise RuntimeError(f"Function.compile() failed! {self.__str__()} ")

        return Function(op, *self.dims, name=f"[{self.__str__()}]")

    def derivative(self, *n) -> Function[_T]: return derivative(self, *n)

    def d(self, n=1) -> Function[_T]: return self.derivative(n)

    def partial_derivative(self, *d) -> Expression[_T]: return partial_derivative(self, *d)

    def pd(self, *d) -> Function[_T]: return self.partial_derivative(*d)

    def antiderivative(self, *d) -> Function[_T]: return antiderivative(self, *d)

    def dln(self) -> Function[_T]: return self.derivative() / self

    def integral(self, *args, **kwargs) -> _T: return self._compile().integral(*args, **kwargs)

    def roots(self, *args, **kwargs) -> _T: return self._compile().roots(*args, **kwargs)


def function_like(y: NumericType, *args: NumericType, **kwargs) -> Function:
    if len(args) == 0 and isinstance(y, Function):
        return y
    else:
        return Function(y, *args, **kwargs)


class derivative(Expression[_T]):
    def __init__(self,   func: Expression, *d, **kwargs):
        super().__init__(None, func, **kwargs)
        self._d = d

    def __str__(self): return f"D_{self._d}({self._children[0].__str__()})"

    def _eval(self, *xargs: NumericType, **kwargs) -> ArrayType[_T] | Expression[_T]:
        return super()._eval(*xargs, **kwargs)


class partial_derivative(Expression[_T]):
    def __init__(self,   func: Expression[_T], *d, **kwargs):
        super().__init__(None, func, **kwargs)
        self._d = d

    def __str__(self): return f"d_{self._d}({self._children[0].__str__()})"

    def _eval(self, *xargs: NumericType, **kwargs) -> ArrayType[_T] | Expression[_T]:
        return super()._eval(*xargs, **kwargs)


class antiderivative(Expression[_T]):
    def __init__(self, func: Expression[_T], *d,  **kwargs):
        super().__init__(None, func, **kwargs)
        self._d = d

    def __str__(self): return f"I_{self._d}({self._children[0].__str__()})"

    def _eval(self, *xargs: NumericType, **kwargs) -> ArrayType[_T] | Expression[_T]:
        return super()._eval(*xargs, **kwargs)
