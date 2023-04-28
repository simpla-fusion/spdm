from __future__ import annotations

import collections
import collections.abc
import dataclasses
import inspect
import typing

import numpy as np

from ..utils.tags import _not_found_, _undefined_, tags

from .Entry import Entry, as_entry, Entry

from .open_entry import open_entry


class Node(object):
    """
    节点类，用于表示数据结构中的节点，节点可以是一个标量（或np.ndarray），也可以是一个列表，也可以是一个字典。
    用于在一般数据结构上附加类型标识（type_hint)。
    """

    _PRIMARY_TYPE_ = (bool, int, float, str, np.ndarray)
    _MAPPING_TYPE_ = dict
    _SEQUENCE_TYPE_ = list

    def __init__(self, d: typing.Any,  parent: typing.Optional[Node] = None, cache=None, appinfo=None, **kwargs) -> None:
        if isinstance(d, Node._PRIMARY_TYPE_):  # 如果 d 是基本类型,  就将其赋值给_cache 属性, 将 None 赋值给 _entry 属性
            self._entry = None
            self._cache = d
        else: # 如果 d 不是基本类型, 就将其赋值给 _entry 属性, 将 None 赋值给 _cache 属性
            self._cache = cache
            self._entry = as_entry(d)

        if self.__class__ is not Node or self._entry is None: #  如果是子类或者 entry 是 None, 就不改变自己的类, 也就是 Node
            pass
        elif self._entry.is_sequence:  # 如果 entry 是列表, 就把自己的类改成列表
            self.__class__ = Node._SEQUENCE_TYPE_
        elif self._entry.is_mapping:  # 如果 entry 是字典, 就把自己的类改成字典
            self.__class__ = Node._MAPPING_TYPE_

        self._parent = parent
        self._appinfo: typing.Mapping[str, typing.Any] = appinfo if appinfo is not None else kwargs

    def _duplicate(self) -> Node:
        other: Node = self.__class__.__new__(self.__class__)
        other._cache = self._cache
        other._entry = self._entry.duplicate() if self._entry is not None else None
        other._parent = self._parent
        other._appinfo = self._appinfo
        return other

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} />"

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}>{self._entry.dump()}</{self.__class__.__name__}>"

    @property
    def annotation(self) -> dict:
        return {"type":  self._entry.__class__.__name__, "appinfo": self._appinfo}

    def __cache__(self) -> typing.Any:
        return self._cache

    def __entry__(self) -> Entry:
        return self._entry

    def __value__(self) -> typing.Any:
        if self._cache is _undefined_:
            self._cache = self._entry.__value__()
        return self._cache

    def _reset(self):
        self._cache = None
        self._entry.reset()

    def _flash(self):
        raise NotImplementedError("flash")

    def _dump(self):
        if self._cache is None:
            self._cache = self._entry.dump()
        else:
            raise NotImplementedError("Merge cache and entry")
        return self._cache

    def __serialize__(self):
        return self._dump()

    def _validate(self, value, type_hint) -> bool:
        if value is _undefined_ or type_hint is _undefined_:
            return False
        else:
            v_orig_class = getattr(value, "__orig_class__", value.__class__)

            if inspect.isclass(type_hint) and inspect.isclass(v_orig_class) and issubclass(v_orig_class, type_hint):
                res = True
            elif typing.get_origin(type_hint) is not None \
                    and typing.get_origin(v_orig_class) is typing.get_origin(type_hint) \
                    and typing.get_args(v_orig_class) == typing.get_args(type_hint):
                res = True
            else:
                res = False
        return res
