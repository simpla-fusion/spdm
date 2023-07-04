from __future__ import annotations

import collections
import collections.abc
import dataclasses
import inspect
import typing
from copy import copy
from types import SimpleNamespace

from ..utils.logger import logger
from ..utils.plugin import Pluggable
from ..utils.tags import _not_found_
from ..utils.typing import (array_type, as_array, is_scalar, HTreeLike, as_value)
from .Path import Path, as_path, PathLike


_T = typing.TypeVar("_T")


class Entry(Pluggable):

    _plugin_registry = {}

    def __init__(self, data:  typing.Any = None, path: Path | None = None, *args,  **kwargs):
        if self.__class__ is Entry:
            entry_type = kwargs.pop("entry_type", None)

            if entry_type is not None:
                super().__dispatch__init__([f"spdm.plugins.data.Plugin{entry_type}#{entry_type}Entry"],
                                           self, data, path, *args, **kwargs)
                return

        self._data = data if data is not _not_found_ else data
        self._path = as_path(path)

    def __copy__(self) -> Entry:
        obj = object.__new__(self.__class__)
        obj._data = self._data
        obj._path = copy(self._path)
        return obj

    def reset(self, value=None, path=None) -> Entry:
        self._data = value
        self._path = as_path(path)
        return self

    def __str__(self) -> str: return f"<{self.__class__.__name__} path=\"{self._path}\" />"

    @property
    def __entry__(self) -> Entry: return self

    @property
    def __value__(self) -> typing.Any:
        if self._data is None or self._data is _not_found_ or len(self._path) > 0:
            self._data = self.query(default_value=_not_found_)
            self._path = Path()
        return self._data

    @property
    def path(self) -> Path: return self._path

    @property
    def is_leaf(self) -> bool: return len(self._path) > 0 and self._path[-1] is None

    @property
    def is_root(self) -> bool: return len(self._path) == 0

    @property
    def is_generator(self) -> bool: return self._path.is_generator

    @property
    def parent(self) -> Entry:
        other = copy(self)
        other._path = self._path.parent
        return other

    def child(self, path=None, *args, **kwargs) -> Entry:
        path = Path(path)
        if len(path) == 0:
            return self

        if self._data is not None or len(self._path) == 0:
            pass
        elif isinstance(self._path[0], (int, slice)):
            self._data = []
        else:
            self._data = {}

        other = copy(self)
        other._path.append(path, *args, **kwargs)
        return other

    def __iter__(self) -> typing.Generator[Entry, None, None]:
        """ Iterate over the children of the Entry."""
        start = None
        while True:
            try:
                value, start = self.find_next(start=start)
            except StopIteration:
                break
            else:
                yield value

    def __next__(self) -> Entry:
        """ Iterate over the slibings of the Entry."""
        if len(self._path) == 0 or not isinstance(self._path[-1], int):
            raise NotImplementedError(f"Can not iterate over {self._path}")

        start = self._path[-1]

        value, idx = self.parent.find_next(start=start)

        if isinstance(value, Entry):
            value = value.__value__

        self._path[-1] = idx

        return value

    def get(self, *args, default_value: typing.Any = ..., **kwargs) -> typing.Any:
        value = self.child(*args, **kwargs).__value__
        if value is _not_found_:
            value = default_value

        if value is Ellipsis:
            raise KeyError(f"Can not find {args} in {self}")

        return value

    def __getitem__(self, *args) -> Entry: return self.child(*args)

    def __setitem__(self, path, value): return self.child(path).update(value)

    def __delitem__(self, *args): return self.child(*args).remove()

    def __equal__(self, other) -> bool:
        if isinstance(other, Entry):
            return other._data == self._data and other._path == self._path
        else:
            return self.query(Path.tags.equal, other)

    @property
    def count(self) -> int: return self.query(Path.tags.count)

    @property
    def exists(self) -> bool: return self.query(Path.tags.exists)

    def dump(self) -> typing.Any: return self.query(Path.tags.dump)

    ###########################################################
    # API: CRUD  operation

    def query(self, op=None, *args, **kwargs) -> typing.Any:
        """
        Query the Entry.
        Same function as `find`, but put result into a contianer.
        Could be overridden by subclasses.
        """      
        return self._path.query(self._data, op, *args, **kwargs)

    def insert(self, *args, **kwargs) -> Entry:
        new_path = self._path.insert(self._data, *args, **kwargs)
        return self.child(new_path)

    def update(self, *args, **kwargs) -> Entry:
        new_path = self._path.update(self._data,  *args, **kwargs)
        return self.child(new_path)

    def find_next(self,  start: PathLike = None, *args, **kwargs) -> typing.Tuple[Entry, PathLike]:
        """
        Find the value from the cache.
        Return a generator of the results.
        Could be overridden by subclasses.
        """
        value, start = self._path.find_next(self._data, start=start, *args, **kwargs)
        return as_entry(value), start

    def remove(self, *args, **kwargs) -> int: return self._path.remove(self._data, *args, **kwargs)

    ###########################################################


def as_entry(obj, *args, **kwargs) -> Entry:
    if isinstance(obj, Entry):
        entry = obj
    elif hasattr(obj.__class__, "__entry__"):
        entry = obj.__entry__
    elif obj is None or obj is _not_found_:
        entry = Entry()
    else:
        entry = Entry(obj, *args, **kwargs)

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
        for e in self._data:
            yield from self._path.find(e, *args, **kwargs)

    def query(self, *args, default_value=None, **kwargs) -> typing.Any:
        """
        Query the Entry.
        Same function as `find`, but put result into a contianer.
        Could be overridden by subclasses.
        """
        res = _not_found_

        for e in self._data:
            res = self._path.query(e, *args, default_value=_not_found_, **kwargs)
            if res is not _not_found_:
                break

        if res is _not_found_:
            res = default_value

        return res

    def insert(self, *args, **kwargs) -> int:
        return self._path.insert(self._data[0], *args, **kwargs)

    def update(self, *args, **kwargs) -> int:
        return self._path.update(self._data[0], *args,   **kwargs)

    def remove(self) -> int:
        return self._path.remove(self._data[0])

    ###########################################################


def deep_reduce(first=None, *others, level=-1):
    if level == 0 or len(others) == 0:
        return first if first is not _not_found_ else None
    elif first is None or first is _not_found_:
        return deep_reduce(others, level=level)
    elif isinstance(first, str) or is_scalar(first):
        return first
    elif isinstance(first, array_type):
        return sum([first, *(v for v in others if (v is not None and v is not _not_found_))])
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
        for k, v in second.items():
            if k not in res:
                res[k] = v
        return res
    elif others[0] is None or others[0] is _not_found_:
        return first
    else:
        raise TypeError(f"Can not merge dict with {others}!")


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
    elif dclass is array_type:
        obj = as_array(obj)
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
