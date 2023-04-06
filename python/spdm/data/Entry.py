from __future__ import annotations

import collections
import collections.abc
import dataclasses
import functools
import inspect
import operator
import typing
from enum import Flag, auto
from types import SimpleNamespace

import numpy as np

from ..common.tags import _not_found_, _undefined_
from ..util.logger import logger
from ..util.misc import serialize
from ..util.sp_export import sp_load_module

from .Path import Path
from .Query import Query

_T = typing.TypeVar("_T")


class Entry(object):
    __slots__ = "_cache", "_path"
    _PRIMARY_TYPE_ = (bool, int, float, str, np.ndarray)

    _registry = {}

    @classmethod
    def register(cls: typing.Type[Entry], name: str):
        """
        Decorator to register a class to the registry.
        """
        def decorator(cls: Entry) -> Entry:
            if isinstance(name, list):
                for n in name:
                    cls._registry[n] = cls
            elif isinstance(name, str):
                cls._registry[name] = cls
            else:
                raise TypeError(f"Invalid name type: {type(name)}")
            return cls
        return decorator

    @classmethod
    def create(cls,  *args, scheme=None, **kwargs):
        """
        Create an entry from a description.
        """
        if isinstance(scheme, str):
            if scheme in cls._registry:
                return cls._registry[scheme](*args, **kwargs)
            else:
                cls_name = f"spdm.plugins.data.Plugin{scheme}#{scheme}Entry"
                n_module = sp_load_module(cls_name)
                return n_module(*args, **kwargs)
        else:
            return Entry(*args, **kwargs)
        # else:
        #     raise TypeError(f"Invalid description type: {type(description)}")

    def __init__(self, cache:   typing.Any = _undefined_, path:  typing.Union[Path, None] = None, **kwargs):
        super().__init__()
        self._cache = cache if cache is not _undefined_ else {}
        self._path: Path = Path(path)

    def duplicate(self) -> Entry:
        obj = object.__new__(self.__class__)
        obj._cache = self._cache
        obj._path = self._path.duplicate()
        return obj

    def reset(self, value=None) -> Entry:
        self._cache = value
        self._path.reset()
        return self

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} cache={type(self._cache)} path={self._path} />"

    @property
    def attribute(self) -> typing.Any:
        return NotImplemented

    @property
    def cache(self) -> typing.Any:
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
    def parent(self) -> Entry:
        other = self.duplicate()
        other._path.pop()
        return other

    def child(self,  *args) -> Entry:
        if len(args) == 0:
            return self
        else:
            other: Entry = self.duplicate()
            other._path.append(*args)
            return other

    def __next__(self) -> typing.Any:
        res, next_index = self.query(op=Query.tags.next)
        if next_index is None:
            raise StopIteration()
        else:
            self._path[-1] = next_index
            return res.get("result", _not_found_)

    def __iter__(self) -> Entry:
        return self.child(Query.tags.first_child)

    def query(self, *args,  **kwargs):
        """
        Query the cache with the given path and arguments.
        @note This method is intended to be overload by subclasses.
        """
        return Query(self._path, *args,   **kwargs).apply(self._cache)

    def get(self, *args, **kwargs) -> typing.Any:
        return self.child(*args).pull(**kwargs)

    def put(self, *args, **kwargs) -> None:

        if len(args) == 1:
            self.push(args[0], **kwargs)
        elif len(args) > 1:
            self.child(args[:-1]).push(args[-1], **kwargs)
        else:
            raise RuntimeError("Can not put empty value!")

    def remove(self, *args) -> Entry:
        return self.child(*args).erase()

    def __getitem__(self, *args) -> Entry:
        return self.child(*args) if len(args) > 0 else self

    def __setitem__(self, *args):
        assert(len(args) > 0)
        return self.child(*args[:-1]).push(args[-1])

    def __delitem__(self, *args):
        return self.child(*args).erase()

    def pull(self, op=Query.tags.read, **kwargs) -> typing.Any:
        return self.query(op=op, **kwargs)

    def push(self, value: typing.Any = None, op=Query.tags.write, **kwargs) -> typing.Any:
        return self.query(value=value,  op=op, **kwargs)

    def setdefault(self, value) -> typing.Any:
        return self.query(op=Query.tags.read, default_value=value, setdefault=True)

    def extend(self, value: typing.Any) -> Entry:
        self.query(value=value, op=Query.tags.extend)
        return self

    def append(self, value: typing.Any) -> Entry:
        self.query(value=value, op=Query.tags.append)
        return self

    def update(self, value: typing.Any) -> Entry:
        self.query(value=value, op=Query.tags.update)
        return self

    def erase(self) -> Entry:
        self.query(op=Query.tags.erase)
        return self

    def count(self) -> int:
        return self.query(op=Query.tags.count)

    def exists(self) -> bool:
        return self.count() > 0

    def equal(self, value) -> bool:
        if isinstance(value, Entry):
            return self._cache is value._cache and self._path == value._path
        else:
            return self.query(value=value, op=Query.tags.equal)

    def dump(self) -> typing.Any:
        return self.__serialize__()

    def dump_named(self) -> typing.Any:
        res = self.dump()
        if isinstance(res, collections.abc.Mapping):
            return SimpleNamespace(**res)
        return res

    def __serialize__(self) -> typing.Any:
        return serialize(self.pull())


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

    def child(self,  *args) -> Entry:
        return EntryChain([e.child(*args) for e in self._cache])

    def push(self,   value: _T, **kwargs) -> _T:
        self._cache[0].child(self._path).push(value, **kwargs)

    def pull(self, default=_undefined_) -> typing.Any:
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

    def child(self,  *args) -> Entry:
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
