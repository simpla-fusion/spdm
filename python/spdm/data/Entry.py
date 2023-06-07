from __future__ import annotations

import collections
import collections.abc
import dataclasses
import functools
import inspect
import operator
import typing
from types import SimpleNamespace
from copy import copy
import numpy as np

from ..utils.logger import logger
from ..utils.misc import serialize
from ..utils.Pluggable import Pluggable
from ..utils.tags import _not_found_
from ..utils.typing import numeric_type, array_type, primary_type
from ..utils.dict_util import reduce_dict
from .Path import Path, as_path

_T = typing.TypeVar("_T")


class Entry(Pluggable):
    __slots__ = "_cache", "_path"

    _plugin_registry = {}

    def __init__(self, cache:  typing.Any = None, path: Path = None, *args, **kwargs):
        if self.__class__ is Entry:
            n_cls_name = kwargs.get("entry_type", None)
            if n_cls_name is not None:
                super().__dispatch__init__([f"spdm.plugins.data.Plugin{n_cls_name}#{n_cls_name}Entry"],
                                           self, cache, path, *args, **kwargs)
                return

        self._cache = cache
        self._path = as_path(path)

    def __copy__(self) -> Entry:
        obj = object.__new__(self.__class__)
        obj._cache = self._cache
        obj._path = copy(self._path)
        return obj

    def reset(self, value=None) -> Entry:
        self._cache = value
        self._path.clear()
        return self

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} cache={type(self._cache)} path={self._path} />"

    @property
    def __entry__(self) -> Entry: return self

    @property
    def __value__(self) -> typing.Any:
        if self._cache is None or len(self._path) > 0:
            self._cache = self.query(default_value=_not_found_)
            self._path = []
        return self._cache

    @property
    def path(self) -> Path: return self._path

    @property
    def is_leaf(self) -> bool: return len(self._path) > 0 and self._path[-1] is None

    @property
    def is_root(self) -> bool: return len(self._path) == 0

    @property
    def parent(self) -> Entry:
        other = copy(self)
        other._path = self._path.parent
        return other

    def child(self, path=None, *args, **kwargs) -> Entry:
        if path is None:
            return self
        else:
            other = copy(self)
            other._path.append(path, *args, **kwargs)
            return other

    @property
    def children(self) -> Entry: return self.child(slice(None))

    def get(self, *args, default_value=_not_found_, **kwargs) -> typing.Any:
        value = self.child(*args).__value__
        if value is _not_found_:
            value = default_value
        return value

    def first_child(self) -> typing.Generator[typing.Any, None, None]:
        yield from self.children().find()

    def __getitem__(self, *args) -> Entry: return self.child(*args)

    def __setitem__(self, path, value): return self.child(path).insert(value)

    def __delitem__(self, *args): return self.child(*args).remove()

    def __equal__(self, other) -> bool:
        if isinstance(other, Entry):
            return other._cache == self._cache and other._path == self._path
        else:
            return self.query({Path.tags.equal: other})

    ###########################################################
    # API: CRUD  operation

    def find(self, *args, **kwargs) -> typing.Generator[typing.Any, None, None]:
        """
        Find the value from the cache.
        Return a generator of the results.
        Could be overridden by subclasses.
        """
        if self._cache is not None and self._cache is not _not_found_:
            yield from self._path.find(self._cache, *args, **kwargs)

    def query(self, *args, default_value=_not_found_, **kwargs) -> typing.Any:
        """
        Query the Entry.
        Same function as `find`, but put result into a contianer.
        Could be overridden by subclasses.
        """
        return self._path.query(self._cache, *args,  default_value=default_value, **kwargs)

    def insert(self, *args, **kwargs) -> int:
        """ Insert """
        return self._path.insert(self._cache, *args, **kwargs)

    def update(self, *args, **kwargs) -> int:
        return self._path.update(self._cache, *args,   **kwargs)

    def remove(self) -> int: return self._path.remove(self._cache)

    ###########################################################

    @property
    def count(self) -> int: return self.query(Path.tags.count)

    @property
    def exists(self) -> bool: return self.count > 0

    def dump(self) -> typing.Any: return serialize(self.query())

    def dump_named(self) -> typing.Any:
        res = self.dump()
        if isinstance(res, collections.abc.Mapping):
            return SimpleNamespace(**res)
        return res


def as_entry(obj) -> Entry:
    if isinstance(obj, Entry):
        entry = obj
    elif hasattr(obj.__class__, "__entry__"):
        entry = obj.__entry__
    elif obj is None or obj is _not_found_:
        entry = None
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


def deep_reduce(first=None, *others, level=-1):
    if level == 0 or len(others) == 0:
        return first if first is not _not_found_ else None
    elif first is None or first is _not_found_:
        return deep_reduce(others, level=level)
    elif isinstance(first, str) or np.isscalar(first):
        return first
    elif isinstance(first, array_type):
        return np.sum([first, *(v for v in others if (v is not None and v is not _not_found_))])
    elif len(others) > 1:
        return deep_reduce(first, deep_reduce(others, level=level), level=level)
    elif others[0] is None or first is _not_found_:
        return first
    elif isinstance(first, collections.abc.Sequence):
        if isinstance(others[0], collections.abc.Sequence) and not isinstance(others, str):
            return [*first, *others[0]]
        else:
            return [*first, others[0]]
    elif isinstance(first, collections.abc.Mapping) and isinstance(others[0], collections.abc.Mapping):
        second = others[0]
        res = {}
        for k, v in first.items():
            res[k] = deep_reduce(v, second.get(k, None), level=level-1)
        for k, v in second.itmes():
            if k not in res:
                res[k] = v
        return res
    elif others[0] is None or others[0] is _not_found_:
        return first
    else:
        raise TypeError(f"Can not merge dict with {others}!")


class CombineEntry(Entry):
    """ CombineEntry is a special Entry that combine multiple Entry into one.    """

    def __init__(self, cache, *args, path=None):
        super().__init__(cache, path=path)
        self._caches = [self._cache, *args]

    def duplicate(self) -> CombineEntry:
        res: CombineEntry = super().duplicate()  # type:ignore
        res._caches = self._caches
        return res

    def child(self, *args, **kwargs) -> CombineEntry:
        other = super().child(*args, **kwargs)
        other._caches = self._caches
        return other

    def find(self, *args, **kwargs) -> typing.Generator[typing.Any, None, None]:
        for e in self._caches:
            e = as_entry(e).child(self._path[:])
            for value in e.find(*args, **kwargs):
                if value is not _not_found_ and value is not None:
                    yield value

    def query(self, *args, default_value=_not_found_, **kwargs):
        values = deep_reduce(* self.find(*args, **kwargs))
        if values is None or values is _not_found_:
            values = default_value
        return values

    def dump(self) -> typing.Any:
        return deep_reduce(* self.find())


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
