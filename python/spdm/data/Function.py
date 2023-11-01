from __future__ import annotations

import collections.abc
import functools
import typing
from copy import copy

import numpy as np

from ..numlib.calculus import (Antiderivative, Derivative, PartialDerivative,
                               find_roots, integral)
from ..numlib.interpolate import interpolate
from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix
from ..utils.numeric import bitwise_and, is_close, meshgrid
from ..utils.tags import _not_found_
from ..utils.typing import (ArrayType, NumericType, array_type, as_array,
                            is_array, numeric_type, scalar_type, get_args, get_origin)
from ..utils.tree_utils import merge_tree_recursive
from .Expression import Expression
from .Functor import Functor, DiracDeltaFun, ConstantsFunc


class Function(Expression):
    """
        Function
        ---------
        A function is a mapping between two sets, the _domain_ and the  _value_.
        The _value_  is the set of all possible outputs of the function.
        The _domain_ is the set of all possible inputs  to the function.

        函数定义域为多维空间时，网格采用rectlinear mesh，即每个维度网格表示为一个数组 _dims_ 。

    """

    def __init__(self, value, *dims, periods=None, _parent=None, **kwargs):
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

        cache = value
        func = None

        if isinstance(cache, (Functor, Expression, np.ufunc)):
            func = cache
            cache = None
        elif callable(cache):
            func = Functor(cache)
            cache = None
        elif cache is not _not_found_:
            cache = as_array(cache)

        self._cache = cache
        self._dims = list(dims) if len(dims) > 0 else None
        self._periods = periods
        self._parent = _parent

        Expression.__init__(self, func, **kwargs)

    # def __str__(self) -> str:
    #     return f"<{self.__class__.__name__}  dims={tuple(self.shape)} {self._func}/>"

    def __copy_from__(self, other: Function) -> Function:
        """ copy from other"""

        Expression.__copy_from__(self, other)

        if isinstance(other, Function):
            self._dims = other._dims
            self._cache = other._cache
            self._periods = other._periods
            self._metadata = other._metadata
            self._parent = other._parent
            return self

    def _repr_svg_(self) -> str:
        try:
            from ..view import View as sp_view
            res = sp_view.plot(self, output="svg")
        except Exception as error:
            # logger.error(error)
            res = None
        return res

    def __serialize__(self) -> typing.Mapping: raise NotImplementedError(f"__serialize__")

    @classmethod
    def __deserialize__(cls, *args, **kwargs) -> Function: raise NotImplementedError(f"__deserialize__")

    def __getitem__(self, idx) -> NumericType:
        if self._cache is None or self._cache is _not_found_:
            return self.__array__()[idx]
        else:
            return self._cache[idx]
        # raise NotImplementedError(f"Function.__getitem__ is not implemented!")

    def __setitem__(self, idx, value) -> None:
        if self._cache is None or self._cache is _not_found_ or len(self._cache.shape) == 0:
            self._cache = self.__array__()

        self._cache[idx] = value
        # raise RuntimeError("Function.__setitem__ is prohibited!")

    @property
    def dims(self) -> typing.List[ArrayType]:
        """ 函数的网格，即定义域的网格 """
        if self._dims is not None:
            return self._dims

        dims = []
        holder = self
        metadata = None
        while hasattr(holder, "_metadata"):
            metadata = holder._metadata

            dims_s, *_ = group_dict_by_prefix(metadata, "coordinate", sep=None)

            if dims_s is not None and len(dims_s) > 0:
                dims_s = {int(k): v for k, v in dims_s.items() if k.isdigit()}
                dims_s = dict(sorted(dims_s.items(), key=lambda x: x[0]))
                for c in dims_s.values():
                    if not isinstance(c, str):
                        d = as_array(c)
                    elif c.startswith("../"):
                        d = as_array(holder._parent.get(c[3:], _not_found_))
                    elif c.startswith(".../"):
                        d = as_array(holder._parent.get(c, _not_found_))
                    elif hasattr(holder.__class__, "get"):
                        d = as_array(holder.get(c, _not_found_))
                    else:
                        d = _not_found_
                    # elif c.startswith("*/"):
                    #     raise NotImplementedError(f"TODO:{self.__class__}.dims:*/")
                    # else:
                    #     d = as_array(holder.get(c, _not_found_))
                    dims.append(d)

            if len(dims) > 0:
                break
            else:
                holder = getattr(holder, "_parent", None)

        if len(dims) == 0 or any([d is _not_found_ for d in dims]):
            raise RuntimeError(f"Can not get dims! {metadata} {dims_s} {holder}")

        elif len(self.periods) > 0:
            dims = [as_array(v) for v in dims]

            periods = self.periods

            for idx in range(len(dims)):
                if periods[idx] is not np.nan and not np.isclose(dims[idx][-1]-dims[idx][0], periods[idx]):
                    raise RuntimeError(
                        f"idx={idx} periods {periods[idx]} is not compatible with dims [{dims[idx][0]},{dims[idx][-1]}] ")
                if not np.all(dims[idx][1:] > dims[idx][:-1]):
                    raise RuntimeError(
                        f"dims[{idx}] is not increasing! {dims[idx][:5]} {dims[idx][-1]} \n {dims[idx][1:] - dims[idx][:-1]}")

        self._dims = dims

        return self._dims

    @property
    def periods(self) -> typing.List[ArrayType]:
        if self._periods is not None:
            return self._periods
        self._periods = self._metadata.get("periods", [])
        return self._periods

    @property
    def ndim(self) -> int: return len(self.dims)
    """ 函数的维度，函数所能接收参数的个数。 """

    @property
    def rank(self) -> int: return 1
    """ 函数的秩，rank=1 标量函数， rank=3 矢量函数 None 待定 """

    @property
    def shape(self) -> typing.List[int]: return [len(d) for d in self.dims]

    @functools.cached_property
    def points(self) -> typing.List[ArrayType]:
        if len(self.dims) == 0:
            raise RuntimeError(self.dims)
        elif len(self.dims) == 1:
            return self.dims
        else:
            return meshgrid(*self.dims, indexing="ij")

    @functools.cached_property
    def bbox(self) -> typing.Tuple[typing.List[float], typing.List[float]]:
        """ 函数的定义域 """
        return tuple(([d[0], d[-1]] if not isinstance(d, float) else [d, d]) for d in self.dims)

    def _type_hint(self, path=None) -> typing.Type:
        tp = get_args(get_origin(self))
        if len(tp) == 0:
            return float
        else:
            return tp[-1]

    def __domain__(self, *args) -> bool:
        # or self._metadata.get("extrapolate", 0) != 1:
        if self.dims is None or len(self.dims) == 0 or self._metadata.get("extrapolate",  0) != "raise":
            return True

        if len(args) != self.ndim:
            raise RuntimeError(f"len(args) != len(self.dims) {len(args)}!={len(self.dims)}")

        v = []
        for i, (xmin, xmax) in enumerate(self.bbox):
            v.append((args[i] >= xmin) & (args[i] <= xmax))

        return bitwise_and.reduce(v)

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

        func = super().__functor__()

        if isinstance(func, (Functor, Expression, np.ufunc)):
            return func

        elif func is not None:
            raise RuntimeError(f"expr is not array_type! {type(func)}")

        dims = self.dims

        value = self._cache

        if value is _not_found_ or value is None:
            self._func = None

        elif isinstance(value, scalar_type):
            self._func = ConstantsFunc(value)

        elif isinstance(value, array_type) and value.size == 1:
            value = np.squeeze(value).item()

            if not isinstance(value, scalar_type):
                raise RuntimeError(f"TODO:  {value}")

            self._func = ConstantsFunc(value)

        elif all([(not isinstance(v, array_type) or v.size == 1) for v in dims]):
            self._func = DiracDeltaFun(value, [float(v) for v in self.dims])

        elif all([(isinstance(v, array_type) and v.ndim == 1 and v.size > 0) for v in dims]):
            self._func = self._interpolate()

        else:
            raise RuntimeError(f"TODO: {dims} {value}")

        return self._func

    def __value__(self, *args, **kwargs) -> array_type | float:
        value = self._cache

        if not isinstance(value, scalar_type) and not isinstance(value, array_type):
            self._cache = self.__call__(*self.dims)
            value = self._cache

        if not isinstance(value, scalar_type) and not isinstance(value, array_type):
            logger.error(f"{self.__class__} \"{(value)}\"")

        return value

    def __array__(self, *args,  **kwargs) -> NumericType:
        """ 重载 numpy 的 __array__ 运算符
                若 self._value 为 array_type 或标量类型 则返回函数执行的结果
        """
        value = self.__value__()

        if isinstance(value, array_type) and len(value.shape) == 0:
            value = np.full(self.shape, value.item())

        return value

    def _interpolate(self):
        value = self.__array__()
        if not isinstance(value, array_type):
            raise RuntimeError(f"self.__array__ is not array_type! {(value)}")
        return interpolate(value, *self.dims,
                           periods=self.periods,
                           extrapolate=self._metadata.get("extrapolate", 0)
                           )

    def __call__(self, *args, **kwargs) -> typing.Any:
        if len(args) == 0 and len(kwargs) == 0:
            return self
        else:
            return super().__call__(*args, **kwargs)

    def derivative(self, *d, **kwargs) -> Function:
        if len(self.__array__().shape) == 0:
            return Function(0.0, self.dims)
        else:
            return Function(self._interpolate().derivative(*d, **kwargs), *self.dims, periods=self.periods, **self._metadata)

    def partial_derivative(self, *d, **kwargs) -> Function:
        return Function(self._interpolate().partial_derivative(*d, **kwargs), *self.dims, periods=self.periods, **self._metadata)

    def antiderivative(self, *d, **kwargs) -> Function:
        return Function(self._interpolate().antiderivative(*d, **kwargs), *self.dims, periods=self.periods, label=rf"\int {self.__repr__()} ")

    def d(self, n=1) -> Expression: return self.derivative(n)

    def pd(self, *d) -> Expression: return self.partial_derivative(*d)

    def dln(self, *args) -> Expression | float:
        if len(args) == 0:
            return self.derivative() / self
        else:
            return self.dln()(*args)

    def integral(self, *args, **kwargs) -> _T: return integral(self, *args, **kwargs)

    def find_roots(self, *args, **kwargs) -> typing.Generator[_T, None, None]:
        yield from find_roots(self, *args, **kwargs)

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

    def validate(self, value=None, strict=False) -> bool:
        """ 检查函数的定义域和值是否匹配 """

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
