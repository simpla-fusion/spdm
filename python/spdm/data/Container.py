from __future__ import annotations

import collections
import collections.abc
import dataclasses
import functools
import inspect
import typing

import numpy as np
from spdm.numlib.misc import array_like

from ..utils.logger import logger
from ..utils.misc import as_dataclass
from ..utils.tags import _not_found_, _undefined_
from .Entry import Entry, as_entry
from .Function import Function
from .Node import Node
from .Path import Path
from .sp_property import sp_property

_TObject = typing.TypeVar("_TObject")


class Container(Node, typing.Container[_TObject]):
    r"""
       Container Node
    """

    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)

    def __serialize__(self) -> dict:
        return self._entry.__serialize__()

    def duplicate(self) -> Container[_TObject]:
        return super().duplicate()  # type:ignore

    # def update(self, value) -> typing.Any:
    #     if isinstance(value, collections.abc.Mapping):
    #         return self._cache.update(value)
    #     else:
    #         raise TypeError(f"Invalid type {type(value)}")

    def get(self, path, default_value=_not_found_, **kwargs):
        return Container._get(self, Path(path), default_value=default_value, **kwargs)

    def __setitem__(self, path, value) -> typing.Any:
        path = Path(path)
        if len(path) == 1:
            parent = self
        else:
            parent = self.get(path[:-1], parents=True)

        # logger.warning("FIXME:当路径中存在 Query时，无法同步 cache 和 entry")

        if isinstance(parent, Container):
            return parent._cache.__setitem__(path[-1], value)
        else:
            return as_entry(parent).insert(path[-1], value)

    def __getitem__(self, path) -> _TObject:
        return Container._get(self, Path(path), default_value=_not_found_)  # type:ignore

    def __delitem__(self, key) -> bool:
        return self._entry.child(key).remove() > 0

    def __contains__(self, key) -> bool:
        return self._entry.child(key).exists

    def __eq__(self, other) -> bool:
        return self._entry.equal(other)

    def __len__(self) -> int:
        return self._entry.count

    def _as_child(self,
                  key: typing.Union[int, str, slice, None],
                  value=_not_found_,
                  type_hint: typing.Type = None,
                  default_value: typing.Any = _not_found_,
                  getter=None,
                  strict=True,
                  **kwargs) -> _TObject:

        # 获得 type_hint
        if isinstance(key, str):
            attr = getattr(self.__class__, key, _not_found_)
            if isinstance(attr, sp_property):  # 若 key 是 sp_property
                if type_hint is None:
                    type_hint = attr.type_hint
                elif type_hint is not attr.type_hint:
                    raise TypeError(f"{type_hint} is not {attr.type_hint}")
                if getter is None:
                    getter = attr.getter
                if default_value is _not_found_:
                    default_value = attr.default_value
                kwargs = {**kwargs, **attr.kwargs}

            if type_hint is None:  # 作为一般属性，由type_hint决定类型
                type_hint = typing.get_type_hints(self.__class__).get(key, None)

        if type_hint is None:  # 由容器类型决定类型
            type_hint = typing.get_args(getattr(self, "__orig_class__", None))
            if len(type_hint) > 0:
                type_hint = type_hint[-1]

        orig_class = type_hint if inspect.isclass(type_hint) else typing.get_origin(type_hint)

        # if value is _not_found_:
        #     # if isinstance(key, str):
        #     #     # 如果 value 为 _not_found_, 则从 cache 中获取
        #     #     value = self._cache.get(key, _not_found_)
        #     if isinstance(key, int) and isinstance(self._cache, collections.abc.Sequence):
        #         if key < len(self._cache):
        #             value = self._cache[key]

        if orig_class is not None and isinstance(value, orig_class):
            # 如果 value 符合 type_hint 则返回之
            return value  # type:ignore

        if value is _not_found_ and key is not None:
            # 如果 value 为 _not_found_, 则从 self._entry 中获取
            value = self._entry.child(key, force=True)

        if getter is not None:  # 若定义 getter
            sig = inspect.signature(getter)
            if len(sig.parameters) == 1:
                value = getter(self)
            else:
                if isinstance(value, Entry):
                    value = value.query(default_value=default_value)

                if value is _not_found_:
                    value = getter(self, None, **kwargs)
                else:
                    value = getter(self, value, **kwargs)

        if orig_class is None:  # 若 type_hint/orig_class 未定义，则由value决定类型
            if isinstance(value, Entry):
                value = value.query(default_value=default_value, **kwargs)
            elif value is _not_found_:
                value = default_value
        elif isinstance(value, orig_class):
            # 如果 value 符合 type_hint 则返回之
            return value  # type:ignore
        elif issubclass(orig_class, Node):  # 若 type_hint 为 Node
            value = type_hint(value, parent=self, **kwargs)
        else:
            if isinstance(value, Entry):
                value = value.query(default_value=default_value, **kwargs)
            elif value is _not_found_:
                value = default_value

            if value is _not_found_:
                pass
            elif isinstance(value, orig_class):
                pass
            elif isinstance(type_hint, Container._PRIMARY_TYPE_):
                value = type_hint(value)
            elif dataclasses.is_dataclass(type_hint):
                value = as_dataclass(type_hint, value)
            elif issubclass(orig_class, np.ndarray):
                value = np.asarray(value)
            else:
                value = type_hint(value, **kwargs)

                # raise TypeError(f"Illegal type hint {type_hint}")

        if strict and inspect.isclass(orig_class) and not isinstance(value, orig_class) and default_value is not _not_found_:
            raise KeyError(f"Can not find {key}! type_hint={type_hint} value={type(value)}")
        # elif isinstance(key, str):
        #     self._cache[key] = value  # type:ignore
        # elif isinstance(key, int):
        #     if not isinstance(self._cache, collections.abc.Sequence):
        #         raise TypeError(
        #             f"Can not set {key}! type_hint={type_hint} value={type(value)} cache={type(self._cache)}")
        #     elif key >= len(self._cache):
        #         self._cache += [_not_found_]*(key+1-len(self._cache))
        #     self._cache[key] = value
        return value  # type:ignore

    @staticmethod
    def _get(obj, path: list, default_value=_not_found_,   **kwargs):

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
                    obj = obj._as_child(None, obj._entry.child(query))
                else:
                    other: Container = obj.duplicate()  # type:ignore
                    other._entry = obj._entry.child(query)
                    obj = other
                continue
            elif isinstance(query,  slice) and isinstance(obj, collections.abc.Sequence):
                obj = obj.duplicate()
                obj._entry = obj._entry.child(query)
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
