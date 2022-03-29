import collections
import collections.abc
import dataclasses
import enum
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
from numpy.lib.function_base import iterable
from spdm.data.normal_util import normal_get

from spdm.logger import logger
from spdm.tags import _not_found_, _undefined_
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

    def __init__(self, cache=_undefined_, path=None, **kwargs):
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
        if len(args) == 0:
            return self
        elif self._path is None or self._path.empty:
            return self.__class__(self._cache, path=Path(*args))
        else:
            return self.__class__(self._cache, path=self._path.duplicate().append(*args))

    def query(self, q: Query) -> Any:
        return normal_filter(self.pull(_not_found_), q)

    def first_child(self) -> Iterator[_TEntry]:
        """
            return next brother neighbor
        """
        d = self.pull(_not_found_)
        if d is _undefined_ or d is _not_found_:
            raise KeyError(self._path)
        elif isinstance(d, collections.abc.Sequence):
            yield from d
        elif isinstance(d, collections.abc.Mapping):
            yield from d.items()
        elif hasattr(d.__class__, "__iter__"):
            yield from d.__iter__()
        else:
            raise NotImplementedError(type(d))

    def pull(self, default=..., set_default=False) -> Any:
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

        obj = normal_get(self._cache, self._path)

        if obj is _not_found_:
            if set_default:
                self._cache = default
                self._path = None
            return default
        else:
            self._cache = obj
            self._path = None
            return obj

    def set_default(self, default) -> Any:
        return self.pull(default, set_default=True)

    class op_tags(Flag):
        null = auto()
        assign = auto()
        extend = auto()
        append = auto()
        update = auto()

    def push(self, value: _T, op: op_tags = op_tags.assign) -> _T:
        if (self._path is None or self._path.empty):
            if self._cache is None or op in (Entry.op_tags.assign, Entry.op_tags.null):
                self._cache = value
                return value
            else:
                return normal_put(self._cache, _undefined_, value, op)
        else:
            if self._cache is not None:
                pass
            elif isinstance(self._path[0], str):
                self._cache = {}
            else:
                self._cache = []

            return normal_put(self._cache, self._path, value, op)

    def extend(self, value: _T) -> _T:
        return self.push(value, Entry.op_tags.extend)

    def append(self, value: _T) -> _T:
        return self.push(value, Entry.op_tags.append)

    def update(self, value: _T) -> _T:
        return self.push(value, Entry.op_tags.update)

    def erase(self) -> None:
        if self._path is None or self._path.empty:
            self._cache = _undefined_
        else:
            normal_erase(self._cache, self._path)
        return

    def count(self) -> int:
        res = self.pull(_not_found_)
        return len(res) if res is not _not_found_ else 0

    def exists(self) -> bool:
        return self.pull(_not_found_) is not _not_found_

    def equal(self, other) -> bool:
        res = self.pull(_not_found_)
        if isinstance(res, Entry) or hasattr(res, "__entry__"):
            raise NotImplementedError((type(res), type(other)))
        return res == other

    def get(self, path, default=...) -> Any:
        return self.child(path).pull(default)

    def put(self, path, value) -> None:
        self.child(path).push(value)

    def dump(self) -> Any:
        return self.__serialize__()

    def __serialize__(self) -> Any:
        if self._path is None or self._path.empty:
            return serialize(self._cache)
        else:
            return serialize(self.pull())


def normal_erase(obj,  path: Path):

    for key in path._items[:-1]:
        try:
            obj = normal_get(obj, key)
        except (IndexError, KeyError):
            return False

    if isinstance(obj, (collections.abc.Mapping, collections.abc.Sequence)) and not isinstance(obj, str):
        del obj[path[-1]]
        return True
    else:
        return False


def normal_put(obj, path, value, op: Entry.op_tags = Entry.op_tags.assign):
    error_message = None

    if isinstance(obj, Entry):
        obj.child(path).push(value, op)
    elif hasattr(obj.__class__, '__entry__'):
        obj.__entry__().child(path).push(value, op)
    elif not isinstance(obj, (collections.abc.MutableMapping, collections.abc.MutableSequence)):
        error_message = f"Can not put value to {type(obj)}!"
    elif path != 0 and not path:  # is None , empty, [],{},""
        if isinstance(obj, collections.abc.Sequence):
            if op in (Entry.op_tags.extend, Entry.op_tags.update, Entry.op_tags.assign):
                if iterable(value) and not isinstance(value, str):
                    obj.extend(value)
                else:
                    error_message = f"{type(value)} is not iterable!"
            elif op in (Entry.op_tags.append):
                obj.append(value)
            else:
                error_message = False
        elif isinstance(obj, collections.abc.Mapping):
            if op in (Entry.op_tags.update, Entry.op_tags.extend, Entry.op_tags.append):
                if isinstance(value, collections.abc.Mapping):
                    for k, v in value.items():
                        normal_put(obj, k, v, op)
                else:
                    error_message = False
            else:
                error_message = False
        else:
            error_message = f"Can not assign value without key!"
    elif isinstance(path, Path) and len(path) == 1:
        normal_put(obj, path[0], value, op)
    elif isinstance(path, Path) and len(path) > 1:
        for idx, key in enumerate(path._items[:-1]):
            if obj in (None, _not_found_, _undefined_):
                error_message = f"Can not put value to {path[:idx]}"
                break
            elif isinstance(obj, Entry) or hasattr(obj, "__entry__"):
                normal_put(obj, path[idx+1:], value, op)
                break
            else:
                next_obj = normal_get(obj, key)
                if next_obj is not _not_found_:
                    obj = next_obj
                else:
                    if isinstance(path._items[idx+1], str):
                        normal_put(obj, key, {}, Entry.op_tags.assign)
                    else:
                        normal_put(obj, key, [], Entry.op_tags.assign)

                    obj = normal_get(obj, key)

        if obj is not _not_found_:
            normal_put(obj, path[-1], value, op)
    elif path in (None, _undefined_):
        if isinstance(obj, collections.abc.MutableMapping) and isinstance(obj, collections.abc.Mapping) \
                and op in (Entry.op_tags.update, Entry.op_tags.extend, Entry.op_tags.append):
            for k, v in value.items():
                normal_put(obj, k, v, op)
        elif op in (Entry.op_tags.update, Entry.op_tags.extend):
            obj.extend(value)
        elif op in (Entry.op_tags.append):
            obj.append(value)
        else:
            error_message = False
    elif isinstance(path, str) and isinstance(obj, collections.abc.Mapping):
        if op is Entry.op_tags.assign:
            obj[path] = value

        else:
            n_obj = obj.get(path, _not_found_)
            if n_obj is _not_found_ or not isinstance(n_obj, (collections.abc.Mapping, collections.abc.MutableSequence, Entry)):
                if op is Entry.op_tags.append:
                    obj[path] = [value]
                else:
                    obj[path] = value
            else:
                normal_put(n_obj, _undefined_, value, op)
    elif isinstance(path, str) and isinstance(obj, collections.abc.MutableSequence):
        for v in obj:
            normal_put(v, path, value, op)
    elif isinstance(path, int) and isinstance(obj, collections.abc.MutableSequence):
        if path < 0 or path >= len(obj):
            error_message = False
        elif op is Entry.op_tags.assign:
            obj[path] = value
        else:
            normal_put(obj[path], _undefined_, value, op)
    elif isinstance(path, slice) and isinstance(obj, collections.abc.MutableSequence):
        if op is Entry.op_tags.assign:
            obj[path] = value
        else:
            for v in obj[path]:
                normal_put(v, _undefined_, value, op)
    elif isinstance(path, collections.abc.Sequence):
        for i in path:
            normal_put(obj, i, value, op)
    elif isinstance(path, collections.abc.Mapping):
        for i, v in path.items():
            normal_put(obj, i, normal_get(value, v), op)
    elif isinstance(path, Query):
        if isinstance(obj, collections.abc.Sequence):
            for idx, v in enumerate(obj):
                if normal_check(v, path):
                    normal_put(obj, idx, value, op)
        else:
            error_message = f"Only list can accept Query as path!"
    else:
        error_message = False

    if error_message is False:
        error_message = f"Illegal operation!"

    if error_message:
        raise RuntimeError(
            f"Operate Error [{op._name_}]:{error_message} [object={type(obj)} key={path} value={type(value)}]")


def normal_get(obj, path):
    if path is None:
        return obj
    elif obj in (None, _not_found_, _undefined_):
        return _not_found_
    elif isinstance(obj, Entry):
        return obj.child(path).pull(_not_found_)
    elif hasattr(obj.__class__, "__entry__"):
        return obj.__entry__().child(path).pull(_not_found_)
    elif isinstance(path, Path):
        for idx, key in enumerate(path._items):
            if isinstance(obj, Entry):
                obj = obj.child(path[idx:]).pull(_not_found_)
                break
            else:
                obj = normal_get(obj, key)

            if obj is _not_found_:
                break
        return obj
    elif isinstance(path, Query):
        obj = [v for idx, v in normal_filter(obj, path)]
        if len(obj) == 0:
            obj = _not_found_
        elif path._only_first:
            obj = obj[0]

        return obj
    elif isinstance(path, (int, slice)) and isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
        return obj[path]
    elif isinstance(path, (int, slice)) and isinstance(obj, collections.abc.Mapping):
        return {k: normal_get(v, path) for k, v in obj.items()}
    elif isinstance(path, str) and isinstance(obj, collections.abc.Mapping):
        return obj.get(path, _not_found_)
    elif isinstance(path, str) and isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
        return [normal_get(v, path) for v in obj]
    elif isinstance(path, set):
        return {k: normal_get(obj, k) for k in path}
    elif isinstance(path, collections.abc.Sequence):
        return [normal_get(obj, k) for k in path]
    elif isinstance(path, collections.abc.Mapping):
        return {k: normal_get(obj, v) for k, v in path.items()}
    elif hasattr(obj, "get") and isinstance(path, str):
        return obj.get(path, _not_found_)
    else:
        raise NotImplementedError(path)


def normal_filter(obj: Sequence, query: Query) -> Iterator[Tuple[int, Any]]:
    only_first = True if not isinstance(query, Query) else query._only_first
    for idx, val in enumerate(obj):
        if normal_check(val, query):
            yield idx, val
            if only_first:
                break


def normal_check(obj, query, expect=None) -> bool:
    if isinstance(query, Query):
        query = query._query

    if query in [_undefined_, None, _not_found_]:
        return obj
    elif isinstance(query, str):
        if query[0] == '$':
            raise NotImplementedError(query)
            # return _op_tag(query, obj, expect)
        elif isinstance(obj, collections.abc.Mapping):
            return normal_get(obj, query) == expect
        elif hasattr(obj, "_entry"):
            return normal_get(obj._entry, query, _not_found_) == expect
        else:
            raise TypeError(query)
    elif isinstance(query, collections.abc.Mapping):
        return all([normal_check(obj, k, v) for k, v in query.items()])
    elif isinstance(query, collections.abc.Sequence):
        return all([normal_check(obj, k) for k in query])
    else:
        raise NotImplementedError(query)


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


def as_entry(obj) -> Entry:

    if isinstance(obj, Entry):
        entry = obj
    elif hasattr(obj.__class__, "__entry__"):
        entry = obj.__entry__()
    else:
        entry = Entry(obj)

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


class EntryCombine(EntryChain):
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
        return EntryCombine([e.child(*args) for e in self._cache],
                            reducer=self._reducer, partition=self._partition)

    def __len__(self):
        raise NotImplementedError()

    def __iter__(self) -> Iterator[Entry]:
        raise NotImplementedError()

    def push(self, *args, **kwargs):
        raise NotImplementedError()

    def pull(self, default: _T = _undefined_, **kwargs) -> _T:
        if not self._path.empty:
            val = EntryCombine([e.child(self._path) for e in self._cache])
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
                val = EntryCombine(val, reducer=self._reducer, partition=self._partition)
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
