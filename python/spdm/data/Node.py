from __future__ import annotations

import collections
import collections.abc
import dataclasses
import inspect
from typing import Any, Callable, TypeVar, Union, get_args, get_origin

import numpy as np

from ..common.tags import _not_found_, _undefined_, tags
from .Entry import Entry, as_entry
from .SpObject import SpObject

_T = TypeVar("_T")
_TKey = TypeVar("_TKey")


class Node(SpObject):
    # __slots__ = "_entry"

    _PRIMARY_TYPE_ = (bool, int, float, str, np.ndarray)
    _MAPPING_TYPE_ = dict
    _SEQUENCE_TYPE_ = list
    _CONTAINER_TYPE_ = None

    def __new__(cls,  *args, **kwargs):
        if cls is not Node:
            obj = object.__new__(cls)
        elif len(args) == 1:
            if isinstance(args[0], collections.abc.Sequence) and not isinstance(args[0], str):
                obj = object.__new__(Node._SEQUENCE_TYPE_)
            elif isinstance(args[0], collections.abc.Mapping):
                obj = object.__new__(Node._MAPPING_TYPE_)
            # elif isinstance(args[0], Entry) and Node._LINK_TYPE_ is not None:
            #     obj = object.__new__(Node._LINK_TYPE_)
            else:
                obj = object.__new__(cls)
        elif len(args) > 1:
            obj = object.__new__(Node._SEQUENCE_TYPE_)
        elif len(kwargs) > 0:
            obj = object.__new__(Node._MAPPING_TYPE_)
        else:
            obj = object.__new__(cls)

        return obj

    def __init__(self, data=None) -> None:
        super().__init__()
        self._entry = as_entry(data)
        self._nid = None
        self._parent = None

    @property
    def annotation(self) -> dict:
        return {"id": self.nid,   "type":  self._entry.__class__.__name__}

    @property
    def nid(self) -> str:
        return self._nid

    @property
    def entry(self) -> Entry:
        return self.__entry__()

    def __entry__(self) -> Entry:
        return self._entry

    def reset(self):
        self._entry.reset()

    def dump(self):
        return self.__serialize__()

    def __serialize__(self):
        return self._entry.dump()

    @property
    def value(self) -> Any:
        return self.update_child(_undefined_, self._entry.pull(_not_found_))

    def _pre_process(self, value: _T, *args, **kwargs) -> _T:
        return value

    def _post_process(self, value: _T, key, *args,  ** kwargs) -> Union[_T, Node]:
        return self.update_child(key, value, *args,  ** kwargs)

    def validate(self, value, type_hint) -> bool:
        if value is _undefined_ or type_hint is _undefined_:
            return False
        else:
            v_orig_class = getattr(value, "__orig_class__", value.__class__)

            if inspect.isclass(type_hint) and inspect.isclass(v_orig_class) and issubclass(v_orig_class, type_hint):
                res = True
            elif get_origin(type_hint) is not None and get_origin(v_orig_class) is get_origin(type_hint) and get_args(v_orig_class) == get_args(type_hint):
                res = True
            else:
                res = False
        return res

    def update_child(self,
                     key: _TKey,
                     value: _T = _undefined_,
                     type_hint=_undefined_,
                     default_value=_undefined_,
                     getter: Callable = _undefined_,
                     in_place=True,
                     force=True,
                     *args, **kwargs) -> Union[_T, Node]:

        is_changed = True

        if value is _undefined_ and key is not _undefined_:
            value = self._entry.child(key).pull(_not_found_)
            is_changed = value is _not_found_

        is_valid = self.validate(value, type_hint) if value is not _not_found_ else False

        if not is_valid:
            if getter is not _undefined_:
                value = getter(self)
            elif value is _undefined_:
                value = default_value
            is_changed = True
            is_valid = self.validate(value, type_hint)

        if is_valid:
            obj = value
        elif type_hint is _undefined_:
            if isinstance(value, (collections.abc.Sequence, collections.abc.Mapping, Entry)) and not isinstance(value, str):
                obj = Node(value, *args, **kwargs)
            else:
                obj = value
            # obj = value if not isinstance(value, Entry) else value.dump()
        elif type_hint in Node._PRIMARY_TYPE_:  # (int, float, bool, str):
            if isinstance(value, Entry):
                value = value.pull(_undefined_)
            elif hasattr(value, "_entry"):
                value = value.value
            if value is _undefined_ or isinstance(value, Entry):
                raise TypeError(value)
            elif type_hint is np.ndarray:
                obj = np.asarray(value)
            elif isinstance(value, tags):
                raise ValueError(f"Tags is not a value! key={key} tags={value}")
            else:
                try:
                    obj = type_hint(value)
                except TypeError as err:
                    raise TypeError(f"Can't convert value {value} to {type_hint}") from err

        elif dataclasses.is_dataclass(type_hint):
            if isinstance(value, collections.abc.Mapping):
                obj = type_hint(**{k: value.get(k, None) for k in type_hint.__dataclass_fields__})
            elif isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
                obj = type_hint(*value)
            else:
                obj = type_hint(value)
        elif inspect.isfunction(type_hint):
            obj = type_hint(value, *args,  **kwargs)
        elif inspect.isclass(type_hint):
            obj = type_hint.__new__(type_hint, value, *args, **kwargs)
            obj._parent = self
            obj.__init__(value, *args, **kwargs)
        elif get_origin(type_hint) is not None:
            obj = type_hint(value, *args, **kwargs)
        else:
            raise NotImplementedError(type_hint)

        # elif hasattr(type_hint, '__origin__'):
            # if issubclass(type_hint.__origin__, Node):
            #     obj = type_hint(value, parent=parent, **kwargs)
            # else:
            #     obj = type_hint(value, **kwargs)
        # if inspect.isclass(type_hint):
        #     if issubclass(type_hint, Node):
        #         obj = type_hint(value, *args, parent=parent, **kwargs)
        # elif callable(type_hint):
        #     obj = type_hint(value, **kwargs)
        # else:
        #     if always_node:
        #         obj = Node(value, *args, parent=parent, **kwargs)
        #     logger.warning(f"Ignore type_hint={type(type_hint)}!")

        is_changed |= obj is not value

        ###################################################################

        if key is not _undefined_ and is_changed:
            if isinstance(obj, Entry) or isinstance(value, Entry):  # and self._entry._cache is value._cache:
                pass
            elif in_place and isinstance(key, (int, str)):
                self._entry.child(key).push(obj)

        if isinstance(obj, Node):
            obj._parent = self

        return obj

        # if isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
        #     res = Node._SEQUENCE_TYPE_(value,  parent=parent, **kwargs)
        # elif isinstance(value, collections.abc.Mapping):
        #     res = Node._MAPPING_TYPE_(value,   parent=parent, **kwargs)
        # elif isinstance(value, Entry):
        #     if Node._LINK_TYPE_ is not None:
        #         res = Node._LINK_TYPE_(value,  parent=parent, **kwargs)
        #     else:
        #         res = Node(value,  parent=parent, **kwargs)
        # if isinstance(value, Node._PRIMARY_TYPE_) or isinstance(value, Node) or value in (None, _not_found_, _undefined_):
        #     return value
