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

    def __init__(self, cache=_undefined_,   path=[], **kwargs):
        super().__init__()
        self._path = path if isinstance(path, Path) else Path(path)
        self._cache = cache
        # if hasattr(cache, "_entry") or isinstance(cache, Entry):
        #     raise RuntimeError((cache))
        # logger.error(cache)

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
        return self.__class__(self._cache, path=self._path.parent)

    def child(self,  *args) -> _TEntry:
        if len(args) == 1 and isinstance(args[0], tuple):
            args = list(args[0])
        return self.__class__(self._cache, path=self._path.duplicate().append(*args))

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
        """ 查找self._cache中位于self._path的值。
            if  value exists then self._cache=value,self._path=[] and return value
            elif default is not _undefined_ return default
            else Entry which is closest to target，



        Args:
            default ([type], optional): [description]. Defaults to _undefined_.
            strict (bool, optional): [description]. Defaults to False.

        Raises:
            KeyError: [description]

        Returns:
            Any: [description]
        """
        if self._path.empty:
            return self._cache

        obj = self._cache
        cachable = True
        for idx, key in enumerate(self._path):
            if isinstance(obj, Entry):
                obj = obj.child(self._path[idx:]).pull(default)
                cachable = False
                break
            elif isinstance(key, Query):
                obj = [v for idx, v in key.filter(obj)]
                cachable = False
            else:
                try:
                    next_obj = Entry.normal_get(obj, key)
                except (IndexError, KeyError):
                    is_found = False

                    if default is _undefined_:
                        obj = Entry(obj, self._path[idx:])
                    else:
                        obj = default
                    break
                else:
                    obj = next_obj

        if cachable and not isinstance(obj, Entry):
            self._cache = obj
            self._path.reset()
        elif hasattr(obj, "_entry"):
            raise RuntimeError(type(obj))
        return obj

    def push(self, value: _T, **kwargs) -> _T:
        if self._path.empty:
            if self._cache is not _undefined_:
                Entry.normal_set(self._cache, _undefined_, value, **kwargs)
            else:
                self._cache = value
            return value

        if self._cache is _undefined_:
            if isinstance(self._path[0], str):
                self._cache = {}
            elif isinstance(self._path[0], (int, slice)):
                self._cache = []
            else:
                raise ValueError(self._path)

        obj = self._cache
        for idx, key in enumerate(self._path[:-1]):
            if isinstance(obj, Entry):
                obj.child(*self.path[idx+1:]).push(value, **kwargs)
                obj = _undefined_
                break
            elif isinstance(key, Query):
                for k, o in key.filter(obj):
                    as_entry(o, *self._path[idx+1:]).push(value, **kwargs)
                obj = _undefined_
                break
            elif not isinstance(obj, collections.abc.Mapping):
                try:
                    obj = self.normal_get(obj, key)
                except (IndexError, KeyError):
                    raise KeyError(self._path[:idx+1])
            elif isinstance(self._path[idx+1], str):
                obj = obj.setdefault(key, {})
            else:
                obj = obj.setdefault(key, [])

        if obj is not _undefined_:
            Entry.normal_set(obj, self._path[-1], value, **kwargs)

        # elif len(self._path) == 1:
        #     Entry.normal_set(self._cache, self._path[0], value, update=update)
        # self._make_parents().push(value, update=update)
        # self._cache = obj
        # self._path = Path(self._path[-1])

        return value

    def erase(self, key=_undefined_) -> bool:
        if self._path.empty:
            self._cache = None
            return

        obj = self._cache

        if key is not _undefined_:
            path = self._path.append(key)
        else:
            path = self._path

        for key in path[:-1]:
            try:
                obj = Entry.normal_get(obj, key)
            except (IndexError, KeyError):
                return False

        if isinstance(obj, (collections.abc.Mapping, collections.abc.Sequence)) and not isinstance(obj, str):
            del obj[path[-1]]
            return True
        else:
            return False

    def count(self) -> int:
        res = self.pull(_not_found_)
        return len(res) if res is not _not_found_ else 0

    def exists(self) -> bool:
        return self.pull(_not_found_) is not _not_found_

    def equal(self, other) -> bool:

        res = self.pull(_not_found_)
        if isinstance(res, Entry) or hasattr(res, "_entry"):
            raise NotImplementedError((type(res), type(other)))
        return res == other

    @staticmethod
    def normal_get(obj, key, default=_not_found_):
        if isinstance(obj, Entry):
            return obj.child(key).pull(default)
        elif key is None:
            return obj
        elif isinstance(key, (int, slice)) and isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
            return obj[key]
        elif isinstance(key, (int, slice)) and isinstance(obj, collections.abc.Mapping):
            return {k: Entry.normal_get(v, key, default) for k, v in obj.items()}
        elif isinstance(key, str) and isinstance(obj, collections.abc.Mapping):
            return obj.get(key, default)
        elif isinstance(key, str) and isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
            return [Entry.normal_get(v, key, default) for v in obj]
        elif isinstance(key, set):
            return {k: Entry.normal_get(obj, k, default) for k in key}
        elif isinstance(key, collections.abc.Sequence):
            return [Entry.normal_get(obj, k, default) for k in key]
        elif isinstance(key, collections.abc.Mapping):
            return {k: Entry.normal_get(obj, v, default) for k, v in key.items()}
        else:
            raise NotImplementedError(key)

    @staticmethod
    def normal_set(obj, key, value, update=False, extend=False):
        if isinstance(obj, Entry):
            raise RuntimeError(obj)
        elif hasattr(obj, '_entry'):
            obj[key] = value
        elif (update or extend) and key is not _undefined_:
            tmp = Entry.normal_get(obj, key, _not_found_)
            if tmp is not _not_found_:
                Entry.normal_set(tmp, _undefined_, value, update=update, extend=extend)
            else:
                Entry.normal_set(obj, key, value)
        elif key is _undefined_:
            if isinstance(obj, collections.abc.Mapping) and isinstance(value, collections.abc.Mapping):
                for k, v in value.items():
                    Entry.normal_set(obj, k, v, update=update)
            elif isinstance(obj, collections.abc.Sequence) and isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
                obj.extend(value)
                # for k, v in enumerate(value):
                #     Entry.normal_set(obj, k, v, update=update)
            else:
                raise KeyError(type(value))
        elif isinstance(key, (int, str, slice)):
            obj[key] = value
        elif isinstance(key, collections.abc.Sequence):
            for i in key:
                Entry.normal_set(obj, i, value, update=update)
        elif isinstance(key, collections.abc.Mapping):
            for i, v in key:
                Entry.normal_set(obj, i, Entry.normal_get(value, v), update=update)
        else:
            raise NotImplemented

    def get(self, path, default=_undefined_) -> Any:
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


def as_entry(obj, *path) -> Entry:

    if isinstance(obj, Entry):
        entry = obj
    elif hasattr(obj, "_entry"):
        entry = obj._entry
    else:
        entry = Entry(obj)
    if len(path) > 0:
        entry = entry.child(*path)

    return entry


class EntryChain(Entry):
    def __init__(self, cache: collections.abc.Sequence,   **kwargs):
        super().__init__([as_entry(a) for a in cache],   **kwargs)

    def child(self,  *args) -> _TEntry:
        return EntryChain([e.child(*args) for e in self._cache])

    def push(self,   value: _T, **kwargs) -> _T:
        self._cache[0].child(self._path).push(value, **kwargs)

    def pull(self, default=_undefined_) -> Any:
        obj = _not_found_
        for e in self._cache:
            obj = e.child(self._path).pull(_not_found_)
            if obj is not _not_found_:
                break
        return obj if obj is not _not_found_ else default


class EntryCombiner(EntryChain):
    def __init__(self,  cache, *args,
                 reducer=_undefined_,
                 partition=_undefined_, **kwargs):
        super().__init__(cache, *args, **kwargs)
        self._reducer = reducer if reducer is not _undefined_ else operator.__add__
        self._partition = partition

    def duplicate(self):
        res = super().duplicate()
        res._reducer = self._reducer
        res._partition = self._partition

        return res

    def child(self,  *args) -> _TEntry:
        return EntryCombiner([e.child(*args) for e in self._cache],
                             reducer=self._reducer, partition=self._partition)

    def __len__(self):
        raise NotImplementedError()

    def __iter__(self) -> Iterator[Entry]:
        raise NotImplementedError()

    def pull(self, default: _T = _undefined_, **kwargs) -> _T:
        if not self._path.empty:
            val = EntryCombiner([e.child(self._path) for e in self._cache])
        else:
            val = []
            type_hint = _undefined_
            for e in self._cache:
                v = e.pull(_not_found_)
                if v is _not_found_:
                    continue
                elif isinstance(v, Entry):
                    raise RuntimeError(v)

                val.append(v)
                type_hint = type(v)

            if len(val) == 0:
                val = default
            elif len(val) == 1:
                val = val[0]
            elif type_hint is np.ndarray:
                val = functools.reduce(self._reducer, np.asarray(val[1:]), np.asarray(val[0]))
            elif type_hint in [int, float, bool, str]:
                val = functools.reduce(self._reducer, val[1:], val[0])
            else:
                val = EntryCombiner(val, reducer=self._reducer, partition=self._partition)
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
