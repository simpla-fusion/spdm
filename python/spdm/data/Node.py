from __future__ import annotations

import collections.abc
import dataclasses
import inspect
import typing
from copy import copy
from enum import Enum

import numpy as np
import pprint

from ..utils.logger import logger
from ..utils.misc import as_dataclass, typing_get_origin
from ..utils.tags import _not_found_, _undefined_
from ..utils.typing import array_type, primary_type
from .Entry import Entry, as_entry
from .Path import Path


class Node:
    """
    节点类，用于表示数据结构中的节点，节点可以是一个标量（或np.ndarray），也可以是一个列表，也可以是一个字典。
    用于在一般数据结构上附加类型标识（type_hint)。
    """

    _MAPPING_TYPE_ = dict
    _SEQUENCE_TYPE_ = list

    def __init__(self, d: typing.Any,
                 parent: Node = None,
                 metadata: typing.Dict[str, typing.Any] = None,
                 **kwargs) -> None:

        if metadata is None:
            metadata = kwargs
            kwargs = {}

        if self.__class__ is not Node or isinstance(d, primary_type) or isinstance(d, Entry):
            pass
        elif isinstance(d, collections.abc.Sequence):  # 如果 entry 是列表, 就把自己的类改成列表
            self.__class__ = Node._SEQUENCE_TYPE_

        elif isinstance(d, collections.abc.Mapping):  # 如果 entry 是字典, 就把自己的类改成字典
            self.__class__ = Node._MAPPING_TYPE_

        self._entry = as_entry(d)
        self._parent = parent
        self._metadata = metadata

        if len(kwargs) > 0:
            logger.warning(f"Ignore kwargs={kwargs}")

    def __serialize__(self) -> typing.Any: return self._entry.dump()

    @classmethod
    def __deserialize__(self, d) -> Node: return Node(d)

    @property
    def __entry__(self) -> Entry: return self._entry

    @property
    def __value__(self) -> typing.Any: return self._entry.__value__

    def __copy__(self) -> Node:
        other: Node = self.__class__.__new__(self.__class__)
        other._entry = copy(self._entry)
        other._metadata = copy(self._metadata)
        other._parent = self._parent
        return other

    def __repr__(self) -> str: return pprint.pformat(self.__serialize__())

    def __str__(self) -> str: return f"{self._entry}"

    def __type_hint__(self) -> typing.Type: return Node

    def __getitem__(self, key) -> typing.Any:
        return as_node(self._entry.child(key), type_hint=self.__type_hint__() or Node, parent=self)

    def __setitem__(self, key, value) -> None: self._entry.child(key).insert(value)

    def __delitem__(self, key) -> bool: return self._entry.child(key).remove() > 0

    def __contains__(self, key) -> bool: return self._entry.child(key).exists

    def __len__(self) -> int: return self._entry.count

    def __iter__(self) -> typing.Generator[typing.Any, None, None]:
        yield from self._entry.find()

    def __equal__(self, other) -> bool: return self._entry.__equal__(other)

    @property
    def _root(self) -> Node | None:
        p = self
        # FIXME: ids_properties is a work around for IMAS dd until we found better solution
        while p._parent is not None and getattr(p, "ids_properties", None) is None:

            p = p._parent
        return p

    def append(self, value) -> Node:
        self._entry.update({Path.tags.append:  value})
        return self

    # def _validate(self, value, type_hint) -> bool:
    #     if value is _undefined_ or type_hint is _undefined_:
    #         return False
    #     else:
    #         v_orig_class = getattr(value, "__orig_class__", value.__class__)

    #         if inspect.isclass(type_hint) and inspect.isclass(v_orig_class) and issubclass(v_orig_class, type_hint):
    #             res = True
    #         elif typing.get_origin(type_hint) is not None \
    #                 and typing.get_origin(v_orig_class) is typing.get_origin(type_hint) \
    #                 and typing.get_args(v_orig_class) == typing.get_args(type_hint):
    #             res = True
    #         else:
    #             res = False
    #     return res

    # def _as_child(self, key: str, value=_not_found_,  *args, **kwargs) -> Node:
    #     raise NotImplementedError("as_child")

    def _find_node_by_path(self, path: str, prefix=None) -> Node:

        if isinstance(prefix, str) and path.startswith(prefix):
            path = path[len(prefix):]

        if isinstance(path, str):
            path = path.split('/')

        if path[0] == '':
            path = path[1:]
            obj = self._get_root()
        else:
            obj = self
        for idx, p in enumerate(path[:]):
            if p is None or not isinstance(obj, Node):
                raise KeyError(f"{path[:idx]} {type(obj)}")
            elif p == '..':
                obj = obj._parent
            elif isinstance(obj, Node):
                obj = obj._as_child(p)

        return obj


def as_node(
    value: typing.Any,
    type_hint: typing.Type = _not_found_,
    default_value=_not_found_,
    metadata=None,
    strict=False,
        **kwargs) -> typing.Any:

    # if parent is not None:
    #     if type_hint is _undefined_:
    #         type_hint = parent.__type_hint__(key)

    #     if default_value is _undefined_:
    #         if isinstance(key, int):
    #             default_value = parent._default_value
    #         elif isinstance(key, str) and isinstance(parent._default_value, collections.abc.Mapping):
    #             default_value = parent._default_value.get(key, None)

    #     if value is _undefined_ or value is _not_found_:
    #         value = parent.get(key, default_value)

    orig_class = typing_get_origin(type_hint)

    if (inspect.isclass(orig_class) and isinstance(value, orig_class)):
        pass
    elif not inspect.isclass(orig_class):  # 如果 type_hint 未定义，则由value决定类型
        if isinstance(value, Entry):
            value = value.query(default_value=default_value)
        elif value is _not_found_:
            value = default_value
    elif issubclass(orig_class, Node):  # 若 type_hint 为 Node
        value = type_hint(value, default_value=default_value, metadata=metadata, **kwargs)
    else:
        if value is _not_found_:
            value = default_value
        elif isinstance(value, Entry):
            value = value.query(default_value=default_value)

        if value is None or value is _not_found_ or isinstance(value, orig_class):
            pass
        elif issubclass(orig_class, np.ndarray):
            value = np.asarray(value)
        elif type_hint in primary_type:
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
            value = type_hint(value)
        else:
            raise TypeError(f"Illegal type hint {type_hint}")

    if strict and inspect.isclass(orig_class) and not isinstance(value, orig_class):
        raise KeyError(f"Can not convert value={type(value)} to  type_hint={type_hint}")
    # elif not isinstance(parent, Node):
    #     pass
    # elif isinstance(parent._cache, collections.abc.MutableMapping):
    #     parent._cache[key] = value

        #     if self._cache is None:
        #         self._cache = {}
        #     self._cache[key] = value
    return value
