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

    def _as_child(self, key: int | slice,  value=_not_found_, *args, default_value=_not_found_, **kwargs) -> _T:
        if self._default_value is not _not_found_:
            # _as_child 中的 default_value 来自 sp_property 的 type_hint， self._default_value 来自 entry,
            # 所以优先采用 self._default_value
            default_value = self._default_value

        if value is _not_found_ or key is not None:
            if isinstance(self._cache, collections.abc.MutableMapping):
                value = self._cache.get(key, _not_found_)

        if key is None or isinstance(key, int):
            n_value = super()._as_child(key, value, *args, default_value=default_value,  **kwargs)
        elif isinstance(key, slice):
            if key.start is None or key.stop is None or key.step is None:
                raise ValueError(f"slice must be a complete slice {key}")
            if isinstance(value, collections.abc.Sequence):
                if len(value) == (key.stop-key.start)/key.step:
                    raise ValueError(f"value must be a sequence with length {(key.stop-key.start)/key.step} {value}")
                n_value = [self._as_child(idx, value[idx], *args, default_value=default_value, **kwargs)
                           for idx in range(key.start, key.stop, key.step)]
            elif isinstance(value, collections.abc.Generator):
                n_value = []
                for idx in range(key.start, key.stop, key.step):
                    n_value.append(self._as_child(idx, next(value), *args, default_value=default_value, **kwargs))
            else:
                raise TypeError(f"key must be int or slice, not {type(key)}")
        else:
            raise RuntimeError(f"Key error ! {key}")

        if isinstance(n_value, Node) and n_value._parent is self:
            n_value._parent = self._parent

        if n_value is not value and n_value is not None and n_value is not _not_found_:
            if self._cache is None:
                self._cache = {}
            self._cache[key] = n_value
        return n_value

    def __iadd__(self, value) -> List:
        self._entry.update({Path.tags.append: value})
        return self

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

    def __getitem__(self, key) -> _T:
        # if isinstance(key, str):
        #     return self._as_child(key, deep_reduce(*super().find({self._unique_id_name: key})))
        # else:
        if not isinstance(key, int):
            raise NotImplementedError(f"key must be int, not {type(key)}")

        value = self._cache.get(key, _not_found_)
        if value is _not_found_:
            value = self._entry.child(key)

        if not isinstance(value, Node):
            n_value = self.as_child(None, value, type_hint=self.__type_hint__(),
                                    default_value=self._default_value, parent=self._parent)
            self._cache[key] = n_value
        else:
            n_value = value

        return n_value


Node._SEQUENCE_TYPE_ = List
