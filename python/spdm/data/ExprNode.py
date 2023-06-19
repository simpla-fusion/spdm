from __future__ import annotations

import collections.abc
import pprint
import typing
from enum import Enum
from functools import cached_property

import numpy as np
from spdm.utils.typing import ArrayType

from ..mesh.Mesh import Mesh
from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix
from ..utils.tags import _not_found_
from ..utils.typing import ArrayLike, ArrayType, array_type, numeric_type
from .Expression import Expression
from .ExprOp import ExprOp
from .Node import Node

_T = typing.TypeVar("_T")


class ExprNode(Expression, Node[_T]):
    """
    Profile
    ---------
    Profile= Function + Node 是具有 Function 特性的 Node。
    Function 的 value 由 Node.__value__ 提供，

    mesh 可以根据 kwargs 中的 coordinate* 从 Node 所在 Tree 获得

    """

    def __init__(self,  value: ArrayLike | Expression, *args,   **kwargs) -> None:
        """
            Parameters
            ----------
            value : ArrayLike
                函数的值
            args : typing.Any
                表达式的参数
            kwargs : typing.Any
                用于传递给 Node 的参数

        """

        if isinstance(value, Expression) or callable(value) or isinstance(value, ExprOp):
            expr = value
            value = None
        else:
            expr = None

        name = kwargs.pop("name", None) or kwargs.get("metadata", {}).get("name", None)

        Expression.__init__(self, expr, *args, name=name)

        Node.__init__(self, value, **kwargs)

        self._value = None

        self._ppoly = None

    def __copy__(self) -> ExprNode:
        """ 复制一个新的 Function 对象 """
        other: ExprNode = Node.__copy__(self)  # type:ignore
        other._op = self._op
        other._name = self._name
        other._children = self._children
        other._ppoly = self._ppoly
        other._value = self._value
        return other

    def _refresh(self):
        if self._value is not None or self._op is not None:
            return
        value = super().__value__
        if value is None and value is _not_found_:
            self._value = None
        elif isinstance(value, Expression) or callable(value) or isinstance(value, ExprOp):
            if self._op is None:
                self._op = value
            else:
                raise RuntimeError(f"Cannot refresh {self} 'op'={self._op} with {value}")
        else:
            self._value = self._normalize_value(value)

    @property
    def __value__(self) -> typing.Any:
        if self._value is None:
            self._refresh()
        return self._value

    def __array__(self, *args,  **kwargs) -> array_type:
        """ 重载 numpy 的 __array__ 运算符
                若 self._value 为 array_type 或标量类型 则返回函数执行的结果
        """
        value = self.__value__

        if (value is None or value is _not_found_):
            if self.callable and hasattr(self, "points"):
                value = self.__call__(*self.points)

        if (value is None or value is _not_found_):
            raise ValueError(f"{self} is None")
            # value = []

        return self._normalize_value(value, *args,  **kwargs)

    def _eval(self, op, *args, **kwargs) -> ArrayLike:
        """ 重载 Expression._eval """
        if op is not None:
            pass
        elif self._ppoly is not None:
            op = self._ppoly
        elif self._op is None:
            op = self._refresh()

        if self._op is None:
            op = self._compile()

        return super()._eval(op, *args, **kwargs)

    @staticmethod
    def _normalize_value(value: ArrayLike, *args, **kwargs) -> ArrayLike:
        """ 将 value 转换为 array_type 类型 """
        if isinstance(value, array_type) or np.isscalar(value):
            pass
        elif value is None or value is _not_found_:
            value = None
        elif isinstance(value, numeric_type):
            value = np.asarray(value, *args, **kwargs)
        elif isinstance(value, tuple):
            value = np.asarray(value, *args, **kwargs)
        elif isinstance(value, collections.abc.Sequence):
            value = np.asarray(value, *args, **kwargs)
        elif isinstance(value, collections.abc.Mapping) and len(value) == 0:
            value = None
        else:
            raise RuntimeError(f"Function._normalize_value() incorrect value {value}! {type(value)}")

        if isinstance(value, array_type) and value.size == 1:
            value = np.squeeze(value).item()

        return value
