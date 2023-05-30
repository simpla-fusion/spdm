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
from ..utils.typing import ArrayType,  numeric_type, ArrayLike
from .Expression import Expression
from .Node import Node
from .ExprOp import ExprOp

_T = typing.TypeVar("_T")


class ExprNode(Expression[_T], Node):
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

    def _refresh(self):
        if self._value is not None or self._op is not None:
            return
        value = Node.__value__(self)
        if value is None and value is _not_found_:
            self._value = None
        elif isinstance(value, Expression) or callable(value) or isinstance(value, ExprOp):
            if self._op is None:
                self._op = value
            else:
                raise RuntimeError(f"Cannot refresh {self} 'op'={self._op} with {value}")
        else:
            self._value = self._normalize_value(value)

    def __duplicate__(self) -> ExprNode:
        """ 复制一个新的 Function 对象 """
        other = Node.__duplicate__(self)
        other._op = self._op
        other._name = self._name
        other._children = self._children
        other._value = self._value
        return other

    @property
    def __value__(self) -> typing.Any:
        if self._value is None:
            self._refresh()
        return self._value

    @property
    def __op__(self) -> typing.Callable:
        if self._ppoly is not None:
            return self._ppoly

        if self._op is None:
            self._refresh()

        if self._op is None:
            return self._compile()
        else:
            return self._op

    def __array__(self, *args,  **kwargs) -> ArrayType:
        """ 重载 numpy 的 __array__ 运算符
                若 self._value 为 array_type 或标量类型 则返回函数执行的结果
        """
        value = self.__value__

        if value is None or value is _not_found_:
            if self.callable:
                value = self._normalize_value(self.__call__(*self.points), *args,  **kwargs)

        return value
