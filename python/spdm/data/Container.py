from __future__ import annotations

import collections
import collections.abc
import dataclasses
import functools
import inspect
import typing
from enum import Enum

import numpy as np
from spdm.data.Node import Node
from spdm.numlib.misc import array_like

from ..utils.logger import logger
from ..utils.misc import as_dataclass, typing_get_origin
from ..utils.tags import _not_found_, _undefined_
from .Entry import Entry, as_entry
from .Node import Node, as_node
from .Path import Path

_T = typing.TypeVar("_T")


class Container(Node, typing.Generic[_T]):
    """
        Container
        ---------
        Container 是所有数据容器的基类，它提供了一些基本的数据容器操作，包括：
        - __getitem__  : 用于获取容器中的数据
        - __setitem__  : 用于设置容器中的数据
        - __delitem__  : 用于删除容器中的数据
        - __contains__ : 用于判断容器中是否存在某个数据
        - __len__      : 用于获取容器中数据的数量
        - __iter__     : 用于迭代容器中的数据

        _TObject 是容器中数据的类型，它可以是任意类型，但是必须是可序列化的。

    """

    def __init__(self, d: typing.Any = None, /,  default_value: typing.Any = None, **kwargs) -> None:
        super().__init__(d,  **kwargs)
        self._default_value = default_value

    def __type_hint__(self) -> typing.Type:
        type_hint = typing.get_args(getattr(self, "__orig_class__", None))
        return type_hint[-1] if len(type_hint) > 0 else None

    def get(self, path, default_value=_not_found_, **kwargs) -> typing.Any:
        value = self._entry.child(path).__value__
        if value is _not_found_ or value is None:
            if default_value is not _not_found_:
                value = default_value
            elif isinstance(path, str) and isinstance(self._default_value, collections.abc.Mapping):
                value = self._default_value[path]
            elif isinstance(path, int):
                value = self._default_value

        return as_node(path, value, type_hint=self.__type_hint__(path), parent=self)

    def clear(self): self._reset()
