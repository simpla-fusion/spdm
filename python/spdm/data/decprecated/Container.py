from __future__ import annotations

import collections
import collections.abc
import dataclasses
import functools
import inspect
import typing
from copy import copy
from enum import Enum

from spdm.data.HTree import HTree
from spdm.numlib.misc import array_like

from ..utils.logger import logger
from ..utils.misc import as_dataclass, typing_get_origin
from ..utils.tags import _not_found_, _undefined_
from .Entry import Entry, as_entry
from .HTree import Node
from .Path import Path

_T = typing.TypeVar("_T")


class Container(HTree[_T]):
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
    pass

    # def get(self, path, default_value=_not_found_, **kwargs) -> Node:

    #     return as_node(self._entry.child(path),type_hint=self.__type_hint__(key),
    #     if value is _not_found_ or value is None:
    #         if default_value is not _not_found_:
    #             value = default_value
    #         elif isinstance(path, str) and isinstance(self._default_value, collections.abc.Mapping):
    #             value = self._default_value[path]
    #         elif isinstance(path, int):
    #             value = self._default_value
    #     return as_node(value, type_hint=self.__type_hint__(path), parent=self)
    # def clear(self): self._reset()
    # def get(self, path, default_value=_not_found_, **kwargs) -> typing.Any:
    #     return Container._get(self, Path(path), default_value=default_value, **kwargs)
