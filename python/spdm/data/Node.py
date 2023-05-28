from __future__ import annotations

import collections.abc
import inspect
import typing

import numpy as np

from ..utils.tags import _undefined_, _not_found_
from ..utils.logger import logger
from .Entry import Entry, as_entry, Entry
from .Path import Path, as_path


class Node:
    """
    节点类，用于表示数据结构中的节点，节点可以是一个标量（或np.ndarray），也可以是一个列表，也可以是一个字典。
    用于在一般数据结构上附加类型标识（type_hint)。
    """

    _PRIMARY_TYPE_ = (bool, int, float, str, np.ndarray)
    _MAPPING_TYPE_ = dict
    _SEQUENCE_TYPE_ = list

    def __init__(self, d: typing.Any = None, *args, cache=None,  parent: typing.Optional[Node] = None,
                 default_value=_not_found_, metadata=None, **kwargs) -> None:
        if len(args) > 0:
            raise RuntimeError(f"Ignore {len(args)} position arguments! [{args}]")
        if isinstance(d, Node._PRIMARY_TYPE_) and cache is None:  # 如果 d 是基本类型,  就将其赋值给_cache 属性, 将 None 赋值给 _entry 属性
            self._entry = None
            self._cache = d

        else:  # 如果 d 不是基本类型, 就将其赋值给 _entry 属性, 将 None 赋值给 _cache 属性
            self._cache = cache
            self._entry = as_entry(d)

        if self.__class__ is not Node or self._entry is None:  # 如果是子类或者 entry 是 None, 就不改变自己的类, 也就是 Node
            pass
        elif self._entry.is_sequence:  # 如果 entry 是列表, 就把自己的类改成列表
            self.__class__ = Node._SEQUENCE_TYPE_
        elif self._entry.is_mapping:  # 如果 entry 是字典, 就把自己的类改成字典
            self.__class__ = Node._MAPPING_TYPE_

        self._parent = parent
        self._default_value = default_value
        self._metadata: typing.Mapping[str, typing.Any] = metadata if metadata is not None else kwargs

    def _duplicate(self) -> Node:
        other: Node = self.__class__.__new__(self.__class__)
        other._cache = self._cache
        other._entry = self.__entry__().duplicate()
        other._parent = self._parent
        other._metadata = self._metadata
        return other

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} />"

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}>{self.__entry__().dump()}</{self.__class__.__name__}>"

    def _annotation(self) -> dict:
        return {"entr_type":  self.__entry__().__class__.__name__, "metadata": dict(self._metadata)}

    def __cache__(self) -> typing.Any:
        return self._cache

    def __entry__(self) -> Entry:
        if self._entry is None:
            raise RuntimeError("No entry!")
        return self._entry

    def __value__(self) -> typing.Any:
        if self._cache is _not_found_ or self._cache is None:
            self._cache = self.__entry__().__value__()
            if self._cache is _not_found_ or self._cache is None:
                self._cache = self._default_value
        return self._cache

    def _reset(self):
        self._cache = None
        if self._entry is not None:
            self._entry.reset()

    def _flash(self): raise NotImplementedError("flash")

    def __serialize__(self): return self.__value__()

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

    def _get_root(self) -> Node:
        p = self
        # FIXME: ids_properties is a work around for IMAS dd until we found better solution
        while p._parent is not None and getattr(p, "ids_properties", None) is None:

            p = p._parent
        return p
