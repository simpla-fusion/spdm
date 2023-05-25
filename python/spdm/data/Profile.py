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
from ..utils.typing import ArrayType, NumericType, numeric_type
from .Container import Container
from .Expression import Expression
from .Function import Function
from .Node import Node

_T = typing.TypeVar("_T")


class Profile(Function[_T], Node):
    """
    Profile
    ---------
    Profile= Function + Node 是具有 Function 特性的 Node。
    Function 的 value 由 Node.__value__ 提供，

    mesh 可以根据 kwargs 中的 coordinate* 从 Node 所在 Tree 获得

    """

    def __init__(self,  value: NumericType | Expression, *args, **kwargs) -> None:
        """
            Parameters
            ----------
            value : NumericType
                函数的值
            dims : typing.Any
                用于坐标轴(dims)，用于构建 Mesh
            kwargs : typing.Any
                    coordinate* : 给出各个坐标轴的path, 用于从 Node 所在 Tree 获得坐标轴(dims)
                    op_*        : 用于传递给运算符的参数
                    *           : 用于传递给 Node 的参数

        """
        if hasattr(value.__class__, "__entry__") or isinstance(value, collections.abc.Mapping):
            entry = value
            value = None
        else:
            entry = None

        mesh, opts, kwargs = group_dict_by_prefix(kwargs, ["mesh", "op"])

        Function.__init__(self, value, *args, mesh=mesh, op=opts)

        Node.__init__(self, entry, **kwargs)

    @property
    def __name__(self) -> str: return self._metadata.get("name", "unnamed")

    @property
    def data(self) -> ArrayType: return self.__array__()

    def _refresh(self, force=False) -> Profile[_T]:
        """ 从 NodeTree 中获取 mesh 和 value """

        if not self.empty and not force:  # 如果已经有值了，就不再刷新
            return

        mesh = self._mesh

        if mesh is None:
            mesh = {}

        if isinstance(self._parent, Node):
            coordinates, *_ = group_dict_by_prefix(self._metadata, "coordinate", sep=None)
            coordinates = {int(k): v for k, v in coordinates.items() if k.isdigit()}
            coordinates = dict(sorted(coordinates.items(), key=lambda x: x[0]))

            if len(coordinates) == 0:
                pass
            elif all([isinstance(c, str) and c.startswith('../grid') for c in coordinates.values()]):
                mesh = getattr(self._parent, "grid", None)
            else:
                mesh["dims"] = tuple([(self._parent._find_node_by_path(c, prefix="../") if isinstance(c, str) else c)
                                      for c in coordinates.values()])

        value = Node.__value__(self)

        if value is None or value is _not_found_:
            value = self._value

        Function.__init__(self, value,  mesh=mesh)

        return self

    def __value__(self) -> ArrayType:
        self._refresh()
        return super().__value__()

    def __call__(self, *args, **kwargs) -> _T | ArrayType:
        self._refresh()
        return super().__call__(*args, **kwargs)

    def derivative(self, n=1) -> Profile[_T]:
        other = super().derivative(n)
        other._parent = self._parent
        return other

    def partial_derivative(self, *d) -> Profile[_T]:
        other = super().partial_derivative(*d)
        other._parent = self._parent
        return other

    def antiderivative(self, *d) -> Profile[_T]:
        other = super().antiderivative(*d)
        other._parent = self._parent
        return other
