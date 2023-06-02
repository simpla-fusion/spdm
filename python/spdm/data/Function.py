from __future__ import annotations

import collections.abc
import functools
import inspect
import typing
from copy import copy
from enum import Enum

import numpy as np

from ..numlib.interpolate import interpolate
from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix, try_get
from ..utils.tags import _not_found_
from ..utils.typing import (ArrayLike, ArrayType, NumericType, array_type,
                            numeric_type, scalar_type)
from .Expression import Expression
from .ExprNode import ExprNode
from .ExprOp import (ExprOp, antiderivative, derivative, find_roots, integral,
                     partial_derivative)
from .Node import Node

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

    def __init__(self, value: ArrayLike | Expression, *dims: ArrayType, periods=None, **kwargs):
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
        super().__init__(value, **kwargs)
        self._dims = [np.asarray(v) for v in dims] if len(dims) > 0 else None
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

    def __duplicate__(self) -> Function:
        """ 复制一个新的 Function 对象 """
        other: Function = super().__duplicate__()
        other._dims = self._dims
        other._periods = self._periods
        return other

    def __serialize__(self) -> typing.Mapping: raise NotImplementedError(f"")

    @classmethod
    def __deserialize__(cls, desc: typing.Mapping) -> Function: raise NotImplementedError(f"")

    @property
    def empty(self) -> bool: return self._value is None and self.dims is None and super().empty

    @property
    def dims(self) -> typing.List[ArrayType]:
        """ for rectlinear mesh 每个维度对应一个一维数组，为网格的节点。"""
        if self._dims is not None:
            return self._dims
        parent = self._parent  # kwargs.get("parent", None)
        metadata = self._metadata  # kwargs.get("metadata", None)
        if isinstance(parent, Node) and isinstance(metadata, collections.abc.Mapping):
            coordinates, *_ = group_dict_by_prefix(metadata, "coordinate", sep=None)
            coordinates = {int(k): v for k, v in coordinates.items() if k.isdigit()}
            coordinates = dict(sorted(coordinates.items(), key=lambda x: x[0]))

            if len(coordinates) > 0:
                self._dims = tuple([(parent._find_node_by_path(c, prefix="../") if isinstance(c, str) else c)
                                    for c in coordinates.values()])
        return self._dims

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
            return self.dims
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

    def __getitem__(self, idx) -> NumericType: raise NotImplementedError(f"Function.__getitem__ is not implemented!")

    def __setitem__(self, *args) -> None: raise RuntimeError("Function.__setitem__ is prohibited!")

    def _compile(self, force=False) -> ExprOp:
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
            return self._ppoly

        self._ppoly = None

        value = self.__value__

        if value is None and self.callable:
            value = self.__call__(*self.points)

        if value is None:
            return None
        elif np.isscalar(value):
            self._ppoly = value
        elif isinstance(value, array_type) and self.dims is not None:
            self._ppoly = interpolate(value, *self.dims, periods=self.periods, name=self._name)
        else:
            raise RuntimeError(f"Illegal value! {type(value)}")

        return self._ppoly

    def compile(self, *args, **kwargs) -> Function:
        return Function(self._compile(*args, **kwargs), *self.dims, name=f"[{self.__str__()}]", periods=self._periods)

    def partial_derivative(self, *d) -> Function[_T]:
        return Function[_T](partial_derivative(self._compile(), *d), *self.dims, periods=self._periods, name=f"d_{list(d)}({self})")

    def antiderivative(self, *d) -> Function[_T]:
        return Function[_T](antiderivative(self._compile(), *d), *self.dims, periods=self._periods, name=f"I_{list(d)}({self})")

    def derivative(self, n=1) -> Function[_T]:
        return Function[_T](derivative(self._compile(), n), name=f"D_{n}({self})")

    def d(self, n=1) -> Function[_T]: return self.derivative(n)

    def pd(self, *d) -> Function[_T]: return self.partial_derivative(*d)

    def dln(self) -> Expression[_T]: return self.derivative() / self

    def integral(self, *args, **kwargs) -> _T: return integral(self._compile(), *args, **kwargs)

    def find_roots(self, *args, **kwargs) -> typing.Generator[_T, None, None]:
        yield from find_roots(self._compile(), *args, **kwargs)


def function_like(y: NumericType, *args: NumericType, **kwargs) -> Function:
    if len(args) == 0 and isinstance(y, Function):
        return y
    else:
        return Function(y, *args, **kwargs)
