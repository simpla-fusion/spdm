from __future__ import annotations

import collections
import collections.abc
import dataclasses
import functools
import inspect
import operator
import typing
from types import SimpleNamespace

import numpy as np

from ..common.Plugin import Pluggable
from ..common.tags import _not_found_
from ..util.logger import logger
from ..util.misc import serialize
from .Path import Path

_T = typing.TypeVar("_T")


class Entry(Pluggable):
    __slots__ = "_cache", "_path"
    _PRIMARY_TYPE_ = (bool, int, float, str, np.ndarray)

    _registry = {}

    @classmethod
    def _guess_plugin_name(cls, *args, **kwargs) -> typing.List[str]:
        n_cls_name = kwargs.get("entry_type", None)
        if n_cls_name is None:
            return []
        else:
            return [f"spdm.plugins.data.Plugin{n_cls_name}#{n_cls_name}Entry"]

    def __new__(cls,  *args, **kwargs):
        if cls is not Entry or "entry_type" not in kwargs:
            return object.__new__(cls)
        else:
            return super().__new__(cls, *args,   **kwargs)

    # @classmethod
    # def register(cls, names: typing.Union[typing.List[str], str], other_cls=None):
    #     """
    #     Decorator to register a class to the registry.
    #     """
    #     if other_cls is not None:
    #         if isinstance(names, str):
    #             cls._registry[names] = other_cls
    #         else:
    #             for n in names:
    #                 cls._registry[n] = other_cls
    #         return other_cls
    #     else:
    #         def decorator(o_cls):
    #             Entry.register(names, o_cls)
    #             return o_cls
    #         return decorator

    # @classmethod
    # def create(cls, *args, scheme=None, **kwargs):
    #     """
    #     Create an entry from a description.
    #     """
    #     if scheme is not None:
    #         try:
    #             res = super().create(scheme, *args, **kwargs)
    #         except ModuleNotFoundError:
    #             pass
    #         else:
    #             return res
    #     else:
    #         return Entry(*args, scheme=scheme,  **kwargs)

    def __init__(self, cache:  typing.Any = None, path: typing.Union[Path, None] = None, **kwargs):
        super().__init__()
        self._cache = cache if cache is not None else {}
        self._path: Path = Path(path) if not isinstance(path, Path) else path.duplicate()

    def duplicate(self) -> Entry:
        obj = object.__new__(self.__class__)
        obj._cache = self._cache
        obj._path = self._path.duplicate()
        return obj

    def reset(self, value=None) -> Entry:
        self._cache = value
        self._path.clear()
        return self

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} cache={type(self._cache)} path={self._path} />"

    def __entry__(self) -> Entry:
        return self

    @property
    def is_sequence(self) -> bool:
        return len(self._path) == 0 and isinstance(self._cache, collections.abc.Sequence)

    @property
    def is_mapping(self) -> bool:
        return len(self._path) == 0 and isinstance(self._cache, collections.abc.Mapping)

    @property
    def cache(self) -> typing.Any:
        return self._cache

    @property
    def path(self) -> Path:
        return self._path

    @property
    def is_leaf(self) -> bool:
        return len(self._path) > 0 and self._path[-1] is None

    @property
    def is_root(self) -> bool:
        return len(self._path) == 0

    @property
    def parent(self) -> Entry:
        other = self.duplicate()
        other._path = self._path.parent
        return other

    def child(self, path=None, *args, **kwargs) -> Entry:
        if path is None:
            return self
        else:
            other = self.duplicate()
            other._path.append(path, *args, **kwargs)
            return other
    # @child.register(set)
    # def _(self, path, *args, **kwargs) -> typing.Mapping[str, typing.Any]:
    #     return {k: self.child(k, *args, **kwargs) for k in path}

    # @child.register(tuple)
    # def _(self, path, *args, **kwargs) -> typing.Tuple[typing.Any]:
    #     if all(isinstance(idx, (slice, int)) for idx in path):
    #         return self._child(path, *args, **kwargs)
    #     else:
    #         return tuple(self.child(k, *args, **kwargs) for k in path)

    def children(self, path=None, *args, **kwargs) -> typing.Any:
        other = self.duplicate()
        other._path.append(slice(None))
        return other

    def first_child(self) -> typing.Generator[typing.Any, None, None]:
        yield from self.children().find()

    def filter(self, **kwargs) -> Entry:
        return self.duplicate().child(kwargs)

    def __getitem__(self, *args) -> Entry:
        return self.child(*args)

    def __setitem__(self, path, value):
        return self.child(path).insert(value)

    def __delitem__(self, *args):
        return self.child(*args).remove()

    # # get value
    # def __next__(self) -> Entry:
    #     return Entry(*self.find(Path.tags.next))

    # def __iter__(self) -> typing.Iterator[Entry]:
    #     return self

    @ property
    def __value__(self):
        return self.query(default_value=_not_found_)

    ###########################################################
    # API: CRUD  operation

    def find(self, *args, **kwargs) -> typing.Generator[typing.Any, None, None]:
        """
        Find the value from the cache.
        Return a generator of the results.
        Could be overridden by subclasses.
        """
        yield from self._path.find(self._cache, *args, **kwargs)

    def query(self, *args, default_value=_not_found_, **kwargs) -> typing.Any:
        """
        Query the Entry.
        Same function as `find`, but put result into a contianer.
        Could be overridden by subclasses.
        """
        return self._path.query(self._cache, *args,  default_value=default_value, **kwargs)

    def insert(self, *args, **kwargs) -> int:
        return self._path.insert(self._cache, *args, **kwargs)

    def update(self, *args, **kwargs) -> int:
        return self._path.update(self._cache, *args,   **kwargs)

    def remove(self) -> int:
        return self._path.remove(self._cache)
    ###########################################################

    def get(self, path, default_value: typing.Any = _not_found_, **kwargs) -> typing.Any:
        res = self.child(path).query(default_value=default_value, **kwargs)
        if res is _not_found_:
            raise IndexError(f"Can not find value at {path}!")
        else:
            return res

    def set(self, path, value, **kwargs) -> typing.Any:
        return self.child(path).insert(value, **kwargs)

    @ property
    def count(self) -> int:
        return self.query(Path.tags.count)

    @ property
    def exists(self) -> bool:
        return self.count > 0

    def equal(self, value) -> bool:
        return self.query(Path.tags.equal, value)

    def __equal__(self, other) -> bool:
        if isinstance(other, Entry):
            return other._cache == self._cache and other._path == self._path
        else:
            return self.equal(other)

    def dump(self) -> typing.Any:
        return self.__serialize__()

    def dump_named(self) -> typing.Any:
        res = self.dump()
        if isinstance(res, collections.abc.Mapping):
            return SimpleNamespace(**res)
        return res

    def __serialize__(self) -> typing.Any:
        return serialize(self.query())

    def combine(self, *args, **kwargs) -> EntryCombine:
        return EntryCombine(self, *args, **kwargs)


def as_entry(obj) -> Entry:
    if isinstance(obj, Entry):
        entry = obj
    elif hasattr(obj.__class__, "__as__entry__"):
        entry = obj.__as__entry__()
    else:
        entry = Entry(obj)
    return entry


class EntryChain(Entry):
    def __init__(self, data_src: typing.List[typing.Any], *args, **kwargs):
        if len(data_src) < 2:
            data_src = [None, *data_src]
        data_src = [as_entry(c) for c in data_src]
        super().__init__(data_src,  *args,  **kwargs)

    ###########################################################
    # API: CRUD  operation

    def find(self, *args, **kwargs) -> typing.Generator[typing.Any, None, None]:
        """
        Find the value from the cache.
        Return a generator of the results.
        Could be overridden by subclasses.
        """
        for e in self._cache:
            yield from self._path.find(e, *args, **kwargs)

    def query(self, *args, default_value=None, **kwargs) -> typing.Any:
        """
        Query the Entry.
        Same function as `find`, but put result into a contianer.
        Could be overridden by subclasses.
        """
        res = _not_found_

        for e in self._cache:
            res = self._path.query(e, *args, default_value=_not_found_, **kwargs)
            if res is not _not_found_:
                break

        if res is _not_found_:
            res = default_value

        return res

    def insert(self, *args, **kwargs) -> int:
        return self._path.insert(self._cache[0], *args, **kwargs)

    def update(self, *args, **kwargs) -> int:
        return self._path.update(self._cache[0], *args,   **kwargs)

    def remove(self) -> int:
        return self._path.remove(self._cache[0])

    ###########################################################


class EntryCombine(Entry):
    def __init__(self, target, *args, common_data={},
                 reducer=None, partition=None, **kwargs):
        super().__init__(common_data, *args, **kwargs)
        self._data_list = as_entry(target).child(slice(None))
        self._reducer = reducer if reducer is not None else operator.__add__
        self._partition = partition

    def duplicate(self) -> EntryCombine:
        res: EntryCombine = super().duplicate()  # type: ignore
        res._data_list = self._data_list
        res._reducer = self._reducer
        res._partition = self._partition

        return res

    # def child(self, *args, **kwargs) -> Entry:
    #     return EntryCombine([e.child(*args, **kwargs) for e in self._cache],
    #                         reducer=self._reducer, partition=self._partition)

    def _reduce(self, val, default_value=None):
        val = [v for v in val if v is not _not_found_ and v is not None]

        if len(val) > 1:
            res = functools.reduce(self._reducer, val[1:], val[0])
        elif len(val) == 1:
            res = val[0]
        else:
            res = default_value
        return res

    def query(self, default_value=_not_found_, **kwargs):
        res = super().query(default_value=_not_found_, **kwargs)

        if res is _not_found_:
            vals = [(v.query(**kwargs) if isinstance(v, Entry) else v)
                    for v in self._data_list.child(self._path[:]).find()]
            res = self._reduce(vals)

        if res is _not_found_:
            res = default_value

        # if res is not _not_found_ and len(kwargs) == 0:
        #     try:
        #         super().insert(res)
        #     except Exception:
        #         logger.debug(super()._path.__repr__)

        return res

    def find(self, *args, **kwargs):
        yield from self._data_list.child(self._path[:]).find()

    def insert(self, *args, **kwargs):

        raise NotImplementedError("EntryCombine does not support insert operation!")

    def update(self, *args, **kwargs) -> int:
        raise NotImplementedError("EntryCombine does not support update operation!")

    def __len__(self):
        raise NotImplementedError()

    def __iter__(self) -> typing.Iterator[Entry]:
        raise NotImplementedError()

    def push(self, *args, **kwargs):
        raise NotImplementedError()

    def pull(self, default: _T = None, **kwargs) -> _T:
        if not self._path.empty:
            val = EntryCombine([e.child(self._path) for e in self._cache])
        else:
            val = []
            type_hint = None
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
    #   obj = None
    elif dclass is np.ndarray:
        obj = np.asarray(obj)
    elif hasattr(obj.__class__, 'get'):
        obj = dclass(**{f.name: as_dataclass(f.type, obj.get(f.name, f.default if f.default is not dataclasses.MISSING else None))
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
