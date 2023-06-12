from __future__ import annotations

import collections.abc
import inspect
import typing

from spdm.utils.tags import _undefined_

from ..utils.logger import logger
from ..utils.misc import serialize
from ..utils.tags import _not_found_, _undefined_
from .Container import Container
from .Entry import CombineEntry, Entry, as_entry, deep_reduce
from .Node import Node
from .Path import Path

_T = typing.TypeVar("_T")


class List(Container[_T], typing.MutableSequence[_T]):
    """
        Sequence
        ---------------
        The collections.abc.Sequence abstract base class defines a much richer interface that goes beyond just 
        __getitem__() 
          __len__(), adding count(), index(), __contains__(), and __reversed__().

    """

    def __init__(self, d=None, *args, ** kwargs):
        super().__init__(d if d is not None else [], *args,   **kwargs)

    def __serialize__(self) -> list: return [serialize(v) for v in self._entry.first_child()]

    # def __getitem__(self, key) -> _T:
    #     value = self._entry.child(key)
    #     return as_node(value, key=key, type_hint=self.__type_hint__(), parent=self._parent)

    def __iter__(self) -> typing.Generator[_T, None, None]:
        type_hint = self.__type_hint__()
        for v in self._entry.child(slice(None)).find():
            yield self.as_child(None, v, type_hint=type_hint, parent=self._parent)

    def insert(self, d, predication=_undefined_, **kwargs) -> int:
        return self._entry.child(predication).update(d, **kwargs)

    def __iadd__(self, value) -> List:
        self._entry.update({Path.tags.append: value})
        return self

    def __len__(self) -> int: return self._entry.count

    def append(self, value) -> List:
        self._entry.update({Path.tags.append: value})
        return self

    def sort(self) -> None:    self._entry.update(Path.tags.sort)

    def find(self, predication, **kwargs) -> typing.Generator[typing.Any, None, None]:
        yield from self._entry.child(predication).find(**kwargs)


class AoS(List[_T]):
    """
        Array of structure
    """

    def __init__(self, *args, id: str = _undefined_, **kwargs):
        super().__init__(*args, **kwargs)
        self._unique_id_name = id if id is not _undefined_ else "$id"
        self._cache = {}

    def combine(self, *args, default_value=None) -> _T:

        d_list = []

        default_value = deep_reduce(default_value, self._default_value)

        if default_value is not None and len(default_value) > 0:
            d_list.append(self._default_value)

        if len(args) > 0:
            d_list.extend(args)

        if self._cache is not None and len(self._cache) > 0:
            raise NotImplementedError(f"NOT IMPLEMENTET YET! {self._cache}")
            # d_list.append(as_entry(self._cache).child(slice(None)))

        if self._entry is not None:
            d_list.append(self._entry.child(slice(None)))

        type_hint = self.__type_hint__()
        return type_hint(CombineEntry({}, *d_list), parent=self._parent)

    def unique_by_id(self, id: str = "$id") -> List[_T]:
        """ 当 element 为 dict时，将具有相同 key=id的element 合并（deep_reduce)
        """
        res = {}
        for d in self.find():
            if not isinstance(d, collections.abc.Mapping):
                raise TypeError(f"{type(d)}")
            key = d.get(id, None)
            res[key] = deep_reduce(res.get(key, None), d)

        return self.__class__([*res.values()], parent=self._parent)

    def as_child(self, key:  int | slice,  value=None, parent=_not_found_, **kwargs) -> _T:
        parent = self._parent if parent is _not_found_ or parent is None else parent
        if isinstance(key, int) and key < 0:
            key = len(self)+key

        # if not isinstance(key, int):
        #     raise NotImplementedError(f"key must be int, not {type(key)}")
        if (value is None or value is _not_found_) and isinstance(key, int):
            value = self._cache.get(key, _not_found_)

        if (value is None or value is _not_found_):
            value = self._entry.child(key)

        value = super().as_child(key, value, parent=parent, **kwargs)

        if isinstance(key, int) and value is not _not_found_:
            self._cache[key] = value

        return value


Node._SEQUENCE_TYPE_ = List
