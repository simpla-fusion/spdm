import collections
import collections.abc
import dataclasses
import functools
import inspect
import operator
from copy import deepcopy
from enum import Enum, Flag, auto
from functools import cached_property
from sys import excepthook
from typing import (Any, Callable, Generic, Iterator, Mapping, Sequence, Tuple,
                    Type, TypeVar, Union)

import numpy as np

from ..common.logger import logger
from ..common.tags import _not_found_, _undefined_
from ..util.dict_util import as_native, deep_merge_dict
from ..util.utilities import serialize
from .Path import Path
from .Query import Query


class EntryTags(Flag):
    append = auto()
    parent = auto()
    next = auto()
    last = auto()


_parent_ = EntryTags.parent
_next_ = EntryTags.next
_last_ = EntryTags.last
_append_ = EntryTags.append

_T = TypeVar("_T")
_TObject = TypeVar("_TObject")
_TPath = TypeVar("_TPath", int,  slice, str, Sequence, Mapping)

_TStandardForm = TypeVar("_TStandardForm", bool, int,  float, str, np.ndarray, list, dict)

_TKey = TypeVar('_TKey', int, str)
_TIndex = TypeVar('_TIndex', int, slice)

_TEntry = TypeVar('_TEntry', bound='Entry')


class Entry(object):
    __slots__ = "_cache", "_path"
    _PRIMARY_TYPE_ = (bool, int, float, str, np.ndarray)

    def __init__(self, cache=None, path=[], **kwargs):
        super().__init__()
        self._path = path if isinstance(path, Path) else Path(path)
        self._cache = cache

    def duplicate(self) -> _TEntry:
        obj = object.__new__(self.__class__)
        obj._cache = self._cache
        obj._path = self._path
        return obj

    def reset(self, value=None) -> _TEntry:
        self._cache = value
        self._path = []
        return self

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} cache={type(self._cache)} path={self._path} />"

    @property
    def cache(self) -> Any:
        return self._cache

    @property
    def path(self) -> Path:
        return self._path

    @property
    def is_leaf(self) -> bool:
        return self._path.is_closed

    @property
    def is_root(self) -> bool:
        return self._path.empty

    @property
    def parent(self) -> _TEntry:
        return self.__class__(self._cache, self._path.parent)

    def child(self, *args) -> _TEntry:
        return self.__class__(self._cache, self._path.duplicate().append(*args))

    def query(self, q: Query) -> Any:
        if not isinstance(q, Query):
            q = Query(q)
        return q.apply(self._cache, self._path)

    def first_child(self) -> Iterator[_TEntry]:
        """
            return next brother neighbor
        """
        d = self.pull()
        if isinstance(d, collections.abc.Sequence):
            yield from d
        elif isinstance(d, collections.abc.Mapping):
            yield from d.items()
        elif hasattr(d.__class__, "__iter__"):
            yield from d.__iter__()
        else:
            raise NotImplementedError(type(d))

    def _make_parents(self) -> _TEntry:
        if len(self._path) == 1:
            if self._cache is not None:
                pass
            elif isinstance(self._path[0], str):
                self._cache = {}
            else:
                self._cache = []
            return self

        obj = self._cache
        for idx, key in enumerate(self._path[:-1]):
            if not isinstance(obj, collections.abc.Mapping) or self._path[idx+1] in obj:
                try:
                    obj = self.normal_get(obj, key)
                except (IndexError, KeyError):
                    raise KeyError(self._path[:idx+1])
            elif isinstance(self._path[idx+1], str):
                obj = obj.setdefault(key, {})
            else:
                obj = obj.setdefault(key, [])
        self._cache = obj
        self._path = Path(self._path[-1])
        return self

    def pull(self, default=_undefined_, strict=False) -> Any:
        if self._path.empty:
            return self._cache

        obj = self._cache

        for idx, key in enumerate(self._path):
            try:
                next_obj = Entry.normal_get(obj, key)
            except (IndexError, KeyError):
                if default is not _undefined_:
                    return default
                else:
                    return Entry(obj, self._path[idx:])

            obj = next_obj

        self._cache = obj
        self._path.reset()

        return obj

    def push(self, value: any, update=False, extend=False) -> None:
        if extend:
            target = self.pull(_not_found_)
            if isinstance(target, collections.abc.Sequence) and not isinstance(target, str):
                target.extend(value)
            else:
                self.push(value)
        elif self._path.empty:
            if update and isinstance(self._cache, collections.abc.Mapping):
                Entry.normal_set(self._cache, None, value, update=update)
            else:
                self._cache = value
        elif len(self._path) == 1:
            Entry.normal_set(self._cache, self._path[0], value, update=update)
        else:
            self._make_parents().push(value, update=update)
        return None

    def erase(self) -> bool:
        if self._path.empty:
            self._cache = None
            return

        obj = self._cache

        for key in self._path[:-1]:
            try:
                obj = Entry.normal_get(obj, key)
            except (IndexError, KeyError):
                return False

        if isinstance(obj, (collections.abc.Mapping, collections.abc.Sequence)) and not isinstance(obj, str):
            del obj[self._path[-1]]

        return True

    def count(self) -> int:
        res = self.pull(_not_found_)
        return len(res) if res is not _not_found_ else 0

    def exists(self) -> bool:
        return self.pull(_not_found_) is not _not_found_

    def equal(self, other) -> bool:
        res = self.pull(_not_found_)
        return res == other

    @staticmethod
    def normal_get(obj, key):
        if isinstance(obj, Entry):
            return obj.child(key).pull()
        elif key is None:
            return obj
        elif isinstance(key, Query):
            return key.apply(obj)
        elif isinstance(key, (int, str, slice)):
            return obj[key]
        elif isinstance(key, set):
            return {k: Entry.normal_get(obj, k) for k in key}
        elif isinstance(key, collections.abc.Sequence):
            return [Entry.normal_get(obj, k) for k in key]
        elif isinstance(key, collections.abc.Mapping):
            return {k: Entry.normal_get(obj, v) for k, v in key.items()}
        else:
            raise NotImplemented(type(key))

    @staticmethod
    def normal_set(obj, key, value, update=True):
        if isinstance(obj, Entry):
            obj.child(key).push(value, update=update)
        elif isinstance(key, (int, str, slice)):
            if not update:
                obj[key] = value
            else:
                try:
                    new_obj = obj[key]
                except (KeyError, IndexError):
                    obj[key] = value
                else:
                    Entry.normal_set(new_obj, None, value, update=True)

        elif key is None:
            if isinstance(value, collections.abc.Mapping):
                for k, v in value.items():
                    Entry.normal_set(obj, k, v, update=update)
            elif isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
                for k, v in enumerate(value):
                    Entry.normal_set(obj, k, v, update=update)
            else:
                raise KeyError(key)
        elif isinstance(key, Query):
            logger.debug(key.apply(obj))
            # for k, o in key.apply(obj):
            #     Entry.normal_set(o, k, value, update=update)
        elif isinstance(key, collections.abc.Sequence):
            for i in key:
                Entry.normal_set(obj, i, value, update=update)
        elif isinstance(key, collections.abc.Mapping):
            for i, v in key:
                Entry.normal_set(obj, i, Entry.normal_get(value, v), update=update)
        else:
            raise NotImplemented

    def get(self, path, default=_undefined_) -> any:
        return self.child(path).pull(default)

    def put(self, path, value) -> None:
        self.child(path).push(value)

    def dump(self) -> Any:
        return self.__serialize__()

    def __serialize__(self) -> Any:
        return serialize(self._cache)


def _slice_to_range(s: slice, length: int) -> range:
    start = s.start if s.start is not None else 0
    if s.stop is None:
        stop = length
    elif s.stop < 0:
        stop = (s.stop+length) % length
    else:
        stop = s.stop
    step = s.step or 1
    return range(start, stop, step)


class EntryCombiner(Entry):
    def __init__(self,  d_list: Sequence = [], /,
                 default_value=_undefined_,
                 reducer=_undefined_,
                 partition=_undefined_, **kwargs):
        super().__init__(default_value, **kwargs)
        self._reducer = reducer if reducer is not _undefined_ else operator.__add__
        self._partition = partition
        self._d_list: Sequence[Entry] = d_list

    def duplicate(self):
        res = super().duplicate()
        res._reducer = self._reducer
        res._partition = self._partition
        res._d_list = self._d_list

        return res

    def __len__(self):
        return len(self._d_list)

    def __iter__(self) -> Iterator[Entry]:
        raise NotImplementedError()

    def replace(self, path, value: _T,   *args, **kwargs) -> _T:
        return super().push(path, value, *args, **kwargs)

    def push(self, path, value: _T,  *args, **kwargs) -> _T:
        path = self._path+Entry.normalize_path(path)
        for d in self._d_list:
            Entry._eval_push(d, path, value, *args, **kwargs)

    def pull(self, **kwargs) -> Any:
        val = super().pull(**kwargs)

        if val is not _not_found_:
            return val

        path = self._path+Entry.normalize_path(path)

        val = []
        for d in self._d_list:
            if isinstance(d, (Entry, EntryContainer)):
                target = Entry._eval_pull(d, path)
                p = None
            else:
                target, p = Entry._eval_path(d, path+[None], force=False)
            if target is _not_found_ or p is not None:
                continue
            target = Entry._eval_filter(target, predication=predication, only_first=only_first)
            if target is _not_found_ or len(target) == 0:
                continue
            val.extend([Entry._eval_pull(d, [], query=query, lazy=lazy) for d in target])

        if len(val) == 0:
            val = _not_found_
        elif len(val) == 1:
            val = val[0]
        elif (inspect.isclass(type_hint) and issubclass(type_hint, EntryContainer)):
            val = EntryCombiner(val)
        elif type_hint in (int, float):
            val = functools.reduce(self._reducer, val[1:], val[0])
        elif type_hint is np.ndarray:
            val = functools.reduce(self._reducer, np.asarray(val[1:]), np.asarray(val[0]))
        else:
            val = EntryCombiner(val)
        # elif any(map(lambda v: not isinstance(v, (int, float, np.ndarray)), val)):
        # else:
        #     val = functools.reduce(self._reducer, val[1:], val[0])

        if val is _not_found_ and lazy is True and query is _undefined_ and predication is _undefined_:
            val = self.duplicate().move_to(path)

        return val


def as_dataclass(dclass, obj, default_value=None):
    if dclass is dataclasses._MISSING_TYPE:
        return obj

    if hasattr(obj, '_entry'):
        obj = obj._entry
    if obj is None:
        obj = default_value

    if obj is None or not dataclasses.is_dataclass(dclass) or isinstance(obj, dclass):
        pass
    # elif getattr(obj, 'empty', False):
    #     obj = None
    elif dclass is np.ndarray:
        obj = np.asarray(obj)
    elif hasattr(obj.__class__, 'get'):
        obj = dclass(**{f.name: as_dataclass(f.type, obj.get(f.name,  f.default if f.default is not dataclasses.MISSING else None))
                        for f in dataclasses.fields(dclass)})
    elif isinstance(obj, collections.abc.Sequence):
        obj = dclass(*obj)
    else:
        try:
            obj = dclass(obj)
        except Exception as error:
            logger.debug((type(obj), dclass))
            raise error
    return obj


def convert_from_entry(cls, obj, *args, **kwargs):
    origin_type = getattr(cls, '__origin__', cls)
    if dataclasses.is_dataclass(origin_type):
        obj = as_dataclass(origin_type, obj)
    elif inspect.isclass(origin_type):
        obj = cls(obj, *args, **kwargs)
    elif callable(cls) is not None:
        obj = cls(obj, *args, **kwargs)

    return obj
