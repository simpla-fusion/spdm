from __future__ import annotations

import collections
import collections.abc
import dataclasses
import functools
import inspect
import typing
from enum import Enum

import numpy as np
from spdm.numlib.misc import array_like

from ..utils.logger import logger
from ..utils.misc import as_dataclass, typing_get_origin
from ..utils.tags import _not_found_, _undefined_
from .Entry import Entry, as_entry
from .Function import Function
from .Node import Node
from .Path import Path


class Container(Node, typing.Container):
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

    def _get_cache(self):
        if self._cache is None:
            self._cache = {}
        return self._cache

    def __setitem__(self, path, value) -> None:
        # logger.warning("FIXME:当路径中存在 Query时，无法同步 cache 和 entry")

        path = Path(path)

        if len(path) == 1:
            if not isinstance(self._get_cache(), (collections.abc.MutableMapping, collections.abc.MutableSequence)):
                raise TypeError(f"{self.__class__.__name__} is not MutableMapping or MutableSequence")
            self._cache[path[0]] = value
        else:
            parent = self.get(path[:-1], parents=True)
            if isinstance(parent, Container):
                parent.__setitem__(path[-1], value)
            else:
                as_entry(parent).insert(path[-1], value)

    def __getitem__(self, path) -> typing.Any:
        return Container._get(self, Path(path), default_value=_not_found_)  # type:ignore

    def __delitem__(self, key) -> bool:
        return self.__entry__().child(key).remove() > 0

    def __contains__(self, key) -> bool:
        return self.__entry__().child(key).exists

    def __len__(self) -> int:
        return self.__entry__().count

    def __iter__(self) -> typing.Generator[typing.Any, None, None]:
        raise NotImplementedError()

    def __eq__(self, other) -> bool:
        return self.__entry__().equal(other)

    def _type_hint(self, *args) -> typing.Type:
        type_hint = typing.get_args(getattr(self, "__orig_class__", None))
        return type_hint[-1] if len(type_hint) > 0 else None

    def _as_child(self,
                  key: typing.Union[int, str, slice, None],
                  value: typing.Any = _not_found_,
                  type_hint: typing.Type = None,
                  strict=False,
                  **kwargs) -> typing.Any:
        if type_hint is None:
            type_hint = self._type_hint(key)
        orig_class = typing_get_origin(type_hint)

        # 如果 value 符合 orig_class 则返回之
        if inspect.isclass(orig_class) and isinstance(value, orig_class):
            return value  # type:ignore
        elif value is _not_found_ and key is not None:
            if isinstance(self._cache, collections.abc.MutableMapping):
                value = self._cache.get(key, _not_found_)

            if value is _not_found_:
                # 如果 value 为 _not_found_, 则从 self.__entry__() 中获取
                value = self.__entry__().child(key, force=True)

        default_value: typing.Any = kwargs.get("default_value", None)

        if default_value is not None and (isinstance(default_value, (collections.abc.Mapping, collections.abc.Sequence)) and len(default_value) > 0):
            if isinstance(value, Entry):
                value = value.query(**kwargs)
            elif value is None or value is _not_found_:
                value = default_value

        if not inspect.isclass(orig_class):  # 若 type_hint/orig_class 未定义，则由value决定类型
            if isinstance(value, Entry):
                value = value.query(**kwargs)
            if value is _not_found_:
                value = default_value
        elif isinstance(value, orig_class):
            # 如果 value 符合 type_hint 则返回之
            return value  # type:ignore
        elif issubclass(orig_class, Node):  # 若 type_hint 为 Node
            value = type_hint(value, parent=self, **kwargs)
        else:
            if isinstance(value, Entry):
                value = value.query(**kwargs)

            if value is _not_found_:
                value = default_value

            if value is None or value is _not_found_ or isinstance(value, orig_class):
                pass
            elif issubclass(orig_class, np.ndarray):
                value = np.asarray(value)
            elif type_hint in Container._PRIMARY_TYPE_:
                value = type_hint(value)
            elif dataclasses.is_dataclass(type_hint):
                value = as_dataclass(type_hint, value)
            elif issubclass(type_hint, Enum):
                if isinstance(value, collections.abc.Mapping):
                    value = type_hint[value["name"]]
                elif isinstance(value, str):
                    value = type_hint[value]
                else:
                    raise TypeError(f"Can not convert {value} to {type_hint}")
            elif callable(type_hint):
                value = type_hint(value, **kwargs)
            else:
                raise TypeError(f"Illegal type hint {type_hint}")

        if strict and inspect.isclass(orig_class) and not isinstance(value, orig_class) and default_value is not None:
            raise KeyError(f"Can not find {key}! type_hint={type_hint} value={type(value)}")
        elif isinstance(key, str) and isinstance(self._cache, collections.abc.MutableMapping):
            self._cache[key] = value

        return value  # type:ignore

    @staticmethod
    def _get(obj, path: list, default_value=_not_found_,   **kwargs) -> typing.Any:

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
                obj = obj._duplicate()
                obj._entry = obj.__entry__().child(query)
                continue
            elif isinstance(query, (str, int)):
                obj = obj._as_child(query, default_value=default_value, **kwargs)
                continue
            else:
                raise TypeError(f"Invalid key type {type(query)}")

        if obj is not _not_found_:
            pass
        elif default_value is _not_found_:
            raise KeyError(f"Key {path} not found")
        else:
            obj = default_value

        return obj

    def get(self, path, default_value=_not_found_, **kwargs) -> typing.Any:
        return Container._get(self, Path(path), default_value=default_value, **kwargs)

    def clear(self):
        self._cache = {}
