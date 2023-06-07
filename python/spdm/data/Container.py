from __future__ import annotations

import collections
import collections.abc
import dataclasses
import functools
import inspect
import typing
from enum import Enum
from copy import copy

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

    def __getitem__(self, key) -> _T:
        return as_node(self._entry.child(key), type_hint=self.__type_hint__(), parent=self)

    def get(self, path, default_value=_not_found_, **kwargs) -> typing.Any:
        value = self._entry.child(path).__value__
        if value is _not_found_ or value is None:
            if default_value is not _not_found_:
                value = default_value
            elif isinstance(path, str) and isinstance(self._default_value, collections.abc.Mapping):
                value = self._default_value[path]
            elif isinstance(path, int):
                value = self._default_value

        return as_node(value, type_hint=self.__type_hint__(path), parent=self)

    def clear(self): self._reset()

    @staticmethod
    def _get(obj: Node, path: list, default_value=_not_found_,  **kwargs) -> typing.Any:

        for idx, query in enumerate(path[:]):
            if not isinstance(obj, Container):
                obj = as_entry(obj).child(path[idx:], force=True).query(defalut_value=default_value, **kwargs)
                break
            elif isinstance(query, set):
                obj = {k: Container._get(obj, [k] + path[idx+1:], **kwargs) for k in query}
                break
            elif isinstance(query, tuple):
                obj = tuple([Container._get(obj, [k] + path[idx+1:], **kwargs) for k in query])
                break
            elif isinstance(query, dict) and isinstance(obj, collections.abc.Sequence):
                only_first = kwargs.get("only_first", False) or query.get("@only_first", True)
                if only_first:
                    obj = obj._as_child(None, obj.__entry__().child(query))
                else:
                    other: Container = obj._duplicate()  # type:ignore
                    other._entry = obj.__entry__().child(query)
                    obj = other
                continue
            elif isinstance(query,  slice) and isinstance(obj, collections.abc.Sequence):
                obj = copy(obj)
                obj._entry = obj.__entry__().child(query)
                continue
            elif isinstance(query, (str, int)):
                obj = obj.get(query, default_value=default_value, **kwargs)
                continue
            else:
                raise TypeError(f"Invalid key type {type(query)}")

        if obj is _not_found_:
            obj = default_value

        return obj

    # def get(self, path, default_value=_not_found_, **kwargs) -> typing.Any:
    #     return Container._get(self, Path(path), default_value=default_value, **kwargs)
