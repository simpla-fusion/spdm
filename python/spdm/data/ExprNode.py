from __future__ import annotations

import typing

from ..utils.typing import (numeric_type, ArrayType, NumericType)
from .ExprOp import ExprOp
from .Expression import Expression

from .HTree import HTree

_T = typing.TypeVar("_T")


class ExprNode(Expression, HTree[_T]):
    """
    ExprNode
    ---------
    ExprNode= Function + Node 是具有 Function 特性的 Node。
    Function 的 value 由 Node.__value__ 提供，

    mesh 可以根据 kwargs 中的 coordinate* 从 Node 所在 Tree 获得

    """

    def __init__(self,  expr: typing.Callable | None, *args, cache: NumericType = None,  **kwargs) -> None:
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

        HTree.__init__(self, cache, **kwargs)

        Expression.__init__(self, expr, *args, name=self.__metadata__.get("name", None))

    def __str__(self): return Expression.__str__(self)

    @property
    def __label__(self) -> str:
        units = self._metadata.get("units", "")
        label = self._metadata.get("label", None) or self._name
        if label is not None:
            return f"{label}  {units}"
        else:
            return units

    def __copy__(self) -> ExprNode:
        """ 复制一个新的 Function 对象 """
        other: ExprNode = HTree.__copy__(self)  # type:ignore
        other._op = self._op
        other._name = self._name
        other._children = self._children

        return other

    def __expr__(self) -> ExprOp | NumericType:
        """ 获取表达式的运算符，若为 constants 函数则返回函数值 """
        expr = super().__expr__()
        if isinstance(expr, ExprOp):
            return expr
        else:
            return self.__value__
