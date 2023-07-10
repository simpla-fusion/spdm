from __future__ import annotations

import collections.abc
import functools
import typing
from copy import copy

import numpy as np

from ..numlib.interpolate import interpolate
from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix
from ..utils.numeric import bitwise_and, is_close, meshgrid
from ..utils.tags import _not_found_
from ..utils.typing import (ArrayLike, ArrayType, NumericType, PrimaryType, normalize_array,
                            array_type, as_array, is_array, is_scalar)
from .Expression import Expression
from .ExprNode import ExprNode
from .ExprOp import (ExprOp, antiderivative, derivative, find_roots, integral,
                     partial_derivative)
from .HTree import HTree

_T = typing.TypeVar("_T")


class Function(ExprNode[_T]):
    """
        Function
        ---------
        A function is a mapping between two sets, the _domain_ and the  _value_.
        The _value_  is the set of all possible outputs of the function.
        The _domain_ is the set of all possible inputs  to the function.

        函数定义域为多维空间时，网格采用rectlinear mesh，即每个维度网格表示为一个数组 _dims_ 。

    """

    def __init__(self, func, *dims, cache: NumericType = None, periods: typing.List[float] = None, **kwargs):
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
            extrapolate: int |str
                控制当自变量超出定义域后的值
                * if ext=0  or 'extrapolate', return the extrapolated value. 等于 定义域无限
                * if ext=1  or 'nan', return nan


        """

        dims = [as_array(v) for v in dims]
        periods = periods if isinstance(periods, collections.abc.Sequence) else [np.nan]*len(dims)

        for idx in range(len(dims)):
            if periods[idx] is not np.nan and not np.isclose(dims[idx][-1]-dims[idx][0], periods[idx]):
                raise RuntimeError(
                    f"idx={idx} periods {periods[idx]} is not compatible with dims [{dims[idx][0]},{dims[idx][-1]}] ")
            if not np.all(dims[idx][1:] > dims[idx][:-1]):
                raise RuntimeError(
                    f"dims[{idx}] is not increasing! {dims[idx][:5]} {dims[idx][-1]} \n {dims[idx][1:] - dims[idx][:-1]}")

        if isinstance(func, array_type):
            cache = func
            func = interpolate(cache, *dims,
                               periods=periods,
                               name=kwargs.get("name", None),
                               extrapolate=kwargs.get("extrapolate", 0))
        elif not callable(func):
            cache = func
            func = None
        else:

            cache = None
            # raise TypeError(f"func is not callable or array_type! {type(func)}")

        super().__init__(func, cache=cache, **kwargs)

        self._dims = dims
        self._periods = periods

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

    def __copy__(self) -> Function:
        """ 复制一个新的 Function 对象 """
        other: Function = super().__copy__()  # type:ignore
        other._dims = self._dims
        other._periods = self._periods
        return other

    def __serialize__(self) -> typing.Mapping: raise NotImplementedError(f"")

    @classmethod
    def __deserialize__(cls, desc: typing.Mapping) -> Function: raise NotImplementedError(f"")

    @property
    def empty(self) -> bool: return len(self.dims) == 0 and super().empty

    @property
    def dims(self) -> typing.List[ArrayType]:
        """ for rectlinear mesh 每个维度对应一个一维数组，为网格的节点。"""
        if len(self._dims) > 0:
            return self._dims
        parent = self._parent  # kwargs.get("parent", None)
        metadata = self._metadata  # kwargs.get("metadata", None)
        if isinstance(parent, HTree) and isinstance(metadata, collections.abc.Mapping):
            coordinates, *_ = group_dict_by_prefix(metadata, "coordinate", sep=None)
            if isinstance(coordinates, collections.abc.Mapping):
                coordinates = {int(k): v for k, v in coordinates.items() if k.isdigit()}
                coordinates = dict(sorted(coordinates.items(), key=lambda x: x[0]))

            if coordinates is not None and len(coordinates) > 0:
                self._dims = [as_array(self.get(c) if isinstance(c, str) else c)
                              for c in coordinates.values()]
        return self._dims

    @property
    def ndim(self) -> int: return len(self.dims)
    """ 函数的维度，即定义域的秩 """

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

    @property
    def shape(self) -> typing.List[int]: return [len(d) for d in self.dims]
    """ 所需数组的形状 """

    @property
    def periods(self) -> typing.List[float]: return self._periods

    @functools.cached_property
    def points(self) -> typing.List[ArrayType]:
        if len(self.dims) == 0:
            raise RuntimeError(self.dims)
        elif len(self.dims) == 1:
            return self.dims
        else:
            return meshgrid(*self.dims, indexing="ij")

    def __domain__(self, *args) -> bool:
        # or self._metadata.get("extrapolate", 0) != 1:
        if len(self.dims) == 0:
            return True

        if len(args) != len(self.dims):
            raise RuntimeError(f"len(args) != len(self.dims) {len(args)}!={len(self.dims)}")

        v = []
        for i, d in enumerate(self.dims):
            if not isinstance(d, array_type):
                v.append(is_close(args[i], d))
            elif len(d.shape) == 0:
                v.append(is_close(args[i], d.item()))
            elif len(d) == 1:
                v.append(is_close(args[i], d[0]))
            else:
                v.append((args[i] >= d[0]) & (args[i] <= d[-1]))
        return bitwise_and.reduce(v)

    def __array__(self, *args,  **kwargs) -> ArrayType:
        """ 重载 numpy 的 __array__ 运算符
                若 self._value 为 array_type 或标量类型 则返回函数执行的结果
        """
        value = self.__value__

        if isinstance(value, array_type) and value.size > 0:
            return value

        self._cache = as_array(self.__call__(*self.points), *args,  **kwargs)

        return self._cache

    def __getitem__(self, idx) -> NumericType: raise NotImplementedError(f"Function.__getitem__ is not implemented!")

    def __setitem__(self, *args) -> None: raise RuntimeError("Function.__setitem__ is prohibited!")

    def __expr__(self) -> ExprOp:
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

        expr = super().__expr__()

        if not isinstance(expr, array_type):
            return expr

        elif expr.size == 1:
            return expr.item()

        else:
            self._op = interpolate(expr, *self._dims,
                                   periods=self._periods,
                                   name=self.__name__,
                                   extrapolate=self._metadata.get("extrapolate", 0))

            return self._op

    def __call__(self, *args, **kwargs) -> typing.Any: return super().__call__(*args, **kwargs)

    def compile(self, *args, **kwargs) -> Function[_T]:
        return Function[_T](interpolate(self.__expr__(), *args,
                                        dims=self.dims,
                                        periods=self._periods, **kwargs),
                            *self.dims, periods=self._periods)

    def partial_derivative(self, *d, **kwargs) -> Function[_T]:
        return Function[_T](partial_derivative(d, self.__expr__(), **kwargs),
                            *self.dims, periods=self._periods)

    def antiderivative(self, *d, **kwargs) -> Function[_T]:

        return Function[_T](antiderivative(d, self.__expr__(), **kwargs),
                            *self.dims, periods=self._periods)

    def derivative(self, *n, **kwargs) -> Function[_T]:
        return Function[_T](derivative(n, self.__expr__(),  **kwargs),
                            *self.dims, periods=self._periods)

    def d(self, n=1) -> Function[_T]: return self.derivative(n)

    def pd(self, *d) -> Function[_T]: return self.partial_derivative(*d)

    def dln(self) -> Expression: return self.derivative() / self

    def integral(self, *args, **kwargs) -> _T:
        expr = self.__expr__()

        if not isinstance(expr, ExprOp):
            raise RuntimeError(f"Function.integral is not implemented for {type(expr)}")

        return integral(expr, *args, **kwargs)

    def find_roots(self, *args, **kwargs) -> typing.Generator[_T, None, None]:
        expr = self.__expr__()
        if not isinstance(expr, ExprOp):
            raise RuntimeError(f"Function.find_roots is not implemented for {type(expr)}")
        yield from find_roots(expr, *args, **kwargs)

    def pullback(self, *dims, periods=None) -> Function:

        other = copy(self)

        if len(dims) != len(self.dims):
            raise RuntimeError(f"len(dims) != len(self._dims) {len(dims)}!={len(self.dims)}")
        new_dims = []
        for idx, d in enumerate(dims):
            if is_array(d) and len(d) == len(self.dims[idx]):
                pass
            elif callable(d):
                d = d(self.dims[idx])
            else:
                raise RuntimeError(f"dims does not match {dims}")

            new_dims.append(d)

        other._dims = new_dims

        return other


def function_like(y: NumericType, *args: NumericType, **kwargs) -> Function:
    if len(args) == 0 and isinstance(y, Function):
        return y
    else:
        return Function(y, *args, **kwargs)
