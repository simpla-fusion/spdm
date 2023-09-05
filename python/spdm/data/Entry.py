from __future__ import annotations

import collections
import collections.abc
import dataclasses
import inspect
import os
import pathlib
import typing
from copy import copy
from functools import reduce

from ..utils.logger import deprecated, logger
from ..utils.plugin import Pluggable
from ..utils.tags import _not_found_, _undefined_
from ..utils.tree_utils import merge_tree_recursive
from ..utils.typing import array_type, as_array, as_value, is_scalar
from ..utils.uri_utils import URITuple, uri_split, uri_split_as_dict
from .Path import Path, PathLike, as_path


class Entry(Pluggable):

    _plugin_registry = {}

    @classmethod
    def __dispatch__init__(cls, name_list, self,  *args, **kwargs) -> None:
        if name_list is None:
            name_list = []

        if len(args) > 0 and isinstance(args[0], (str, URITuple)):
            scheme = uri_split(args[0]).scheme

            if scheme in ["file", "local", "https", "http", "", None]:
                raise NotImplementedError(f"")
            else:
                name_list.append(EntryProxy)

        if name_list is None or len(name_list) == 0:
            return super().__init__(self,  *args, **kwargs)
        else:
            return super().__dispatch__init__(name_list, self, *args, **kwargs)

    def __init__(self, data:  typing.Any = None, path: Path | PathLike = None, *args,  **kwargs):
        if self.__class__ is Entry and isinstance(data, (str, URITuple)):
            Entry.__dispatch__init__(None, self, data, path, *args, **kwargs)
            return

        self._data = data
        self._path = as_path(path)

    def __copy__(self) -> Entry:
        obj = object.__new__(self.__class__)
        obj.__copy_from__(self)
        return obj

    def __copy_from__(self, other: Entry) -> Entry:
        self._data = other._data
        self._path = copy(other._path)
        return self

    def reset(self, value=None, path=None) -> Entry:
        self._data = value
        self._path = as_path(path)
        return self

    def __str__(self) -> str: return f"<{self.__class__.__name__} path=\"{self._path}\" />"

    def __getitem__(self, *args) -> Entry: return self.child(*args)

    def __setitem__(self, path, value): return self.child(path).update(value)

    def __delitem__(self, *args): return self.child(*args).remove()

    @property
    def __entry__(self) -> Entry: return self

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
        elif isinstance(self._path[0], str):
            self._data = {}
        else:
            self._data = []

        other = copy(self)
        other._path.append(path)
        return other

    ###########################################################

    @property
    def __value__(self) -> typing.Any: return self._data if len(self._path) == 0 else self.get()

    def get(self, query=None, default_value: typing.Any = _not_found_, **kwargs) -> typing.Any:
        if query is None:
            entry = self
            args = ()
        elif isinstance(query, (slice, set, dict)):
            entry = self
            args = (query,)
        else:
            entry = self.child(query)
            args = ()

        return entry.fetch(Path.tags.fetch, *args, default_value=default_value, **kwargs)

    def dump(self) -> typing.Any: return self.fetch(Path.tags.dump)

    def equal(self, other) -> bool:
        if isinstance(other, Entry):
            return self.fetch(Path.tags.equal, other.__value__)
        else:
            return self.fetch(Path.tags.equal, other)

    @property
    def count(self) -> int: return self.fetch(Path.tags.count)

    @property
    def exists(self) -> bool: return self.fetch(Path.tags.exists)

    def check_type(self, tp: typing.Type) -> bool: return self.fetch(Path.tags.check_type, tp)

    ###########################################################
    # API: CRUD  operation

    def insert(self, value,  **kwargs) -> Entry:
        self._data, next_path = self._path.insert(self._data,  value,  **kwargs)
        return self.__class__(self._data, next_path)

    def update(self, value,   **kwargs) -> Entry:
        self._data = self._path.update(self._data, value, **kwargs)
        return self

    def remove(self,  **kwargs) -> int:
        self._data, num = self._path.remove(self._data,  **kwargs)
        return num

    def fetch(self, op=None, *args, **kwargs) -> typing.Any:
        """
            Query the Entry.
            Same function as `find`, but put result into a contianer.
            Could be overridden by subclasses.
        """
        return self._path.fetch(self._data, op, *args, **kwargs)

    def for_each(self, *args, **kwargs) -> typing.Generator[typing.Tuple[int, typing.Any], None, None]:
        """ Return a generator of the results. """
        yield from self._path.for_each(self._data, *args, **kwargs)

    @deprecated
    def find_next(self, start: int | None, *args, **kwargs) -> typing.Tuple[typing.Any, int | None]:
        """
            Find the value from the cache.
            Return a generator of the results.
            Could be overridden by subclasses.
        """
        return self._path.find_next(self._data, start, *args,  **kwargs)

    ###########################################################


def open_entry(url: str, *args, schema=None, ** kwargs) -> Entry:
    return Entry(url, *args, schema=schema, **kwargs)


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


def convert_from_entry(cls, obj, *args, **kwargs):
    origin_type = getattr(cls, '__origin__', cls)
    if dataclasses.is_dataclass(origin_type):
        obj = as_dataclass(origin_type, obj)
    elif inspect.isclass(origin_type):
        obj = cls(obj, *args, **kwargs)
    elif callable(cls) is not None:
        obj = cls(obj, *args, **kwargs)

    return obj


SPDB_XML_NAMESPACE = "{http://fusionyun.org/schema/}"
SPDB_TAG = "spdb"


class EntryProxy(Entry):

    _maps = None
    _mapping_path = []

    @classmethod
    def load_mappings(cls,
                      mapping_path: typing.List[str] | str | None = None,
                      default_source_schema: str = "EAST",
                      default_target_schema: str = "imas/3",
                      **kwargs,
                      ):

        if isinstance(mapping_path, str):
            mapping_path = mapping_path.split(":")
        elif isinstance(mapping_path, pathlib.Path):
            mapping_path = [mapping_path]
        elif mapping_path is None:
            mapping_path = []

        mapping_path += os.environ.get("SP_DATA_MAPPING_PATH", "").split(":")

        mapping_path = [pathlib.Path(p) for p in mapping_path if p != ""]

        if len(mapping_path) == 0:
            raise RuntimeError(f"No mapping file!  SP_DATA_MAPPING_PATH={os.environ.get('SP_DATA_MAPPING_PATH', '')}")

        cls._default_source_schema: str = default_source_schema
        cls._default_target_schema: str = default_target_schema

        cls._envs = merge_tree_recursive(kwargs.pop("envs", {}), kwargs)

        cls._maps = {}

    @classmethod
    def find_map(cls, source_schema:  str, target_schema:  str, *args, **kwargs) -> Entry:
        if cls._maps is None:
            cls.load_mappings()

        if source_schema is None:
            source_schema = cls._default_source_schema
        if target_schema is None:
            target_schema = cls._default_target_schema

        if source_schema == target_schema:
            logger.debug(f"Source and target schema are the same! {source_schema}")
            return None

        map_tag = f"{source_schema}/{target_schema}"

        mapper = cls._maps.get(map_tag, _not_found_)

        if mapper is _not_found_:

            file_path_suffix = ["config.xml", "static/config.xml", "dynamic/config.xml"]

            mapping_files: typing.List[str] = []
            for m_dir in EntryProxy._mapping_path:
                if not m_dir:
                    continue
                elif isinstance(m_dir, str):
                    m_dir = pathlib.Path(m_dir)
                for file_name in file_path_suffix:
                    p = m_dir / map_tag / file_name
                    if p.exists():
                        mapping_files.append(p)

            if len(mapping_files) == 0:
                raise FileNotFoundError(f"Can not find mapping files for {map_tag}!")

            mapper = File(mapping_files, mode="r", format="XML").read()

            cls._maps[map_tag] = mapper

        return mapper

    def __init__(self, entry: str | URITuple | Entry, *args, schema: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)

        self._mapper = mapper
        self._entry = {}

        spdb_conf = self._mapper.child("spdb").fetch()

        prefix = kwargs.get("prefix", None) or spdb_conf.get("@prefix", None) or os.environ.get("SPDB_PREFIX", None)

        for entry in spdb_conf.get("entry", []):
            id = entry.get("id", None)
            if id is None:
                continue
            url = entry.get("_text", "").format(prefix=prefix)

            self._entry[id] = url

    def __copy__(self) -> Entry:
        obj = object.__new__(self.__class__)
        obj.__copy_from__(self)
        obj._mapper = self._mapper
        return obj

    def insert(self, value, **kwargs) -> Entry: raise NotImplementedError(f"")

    def update(self, value, **kwargs) -> Entry: raise NotImplementedError(f"")

    def remove(self, **kwargs) -> int: raise NotImplementedError(f"")

    def fetch(self, *args, default_value=_not_found_, **kwargs) -> typing.Any:
        request = self._mapper.child(self._path).fetch(
            *args,
            default_value=_not_found_,
            lazy=False,
            **kwargs
        )

        return self._op_fetch(request, default_value=default_value)

    def for_each(self, *args, **kwargs) -> typing.Generator[typing.Tuple[int, typing.Any], None, None]:
        """Return a generator of the results."""
        for idx, request in self._mapper.child(self._path).for_each(*args, **kwargs):
            yield idx, self._op_fetch(request)

    def _op_fetch(self, request: typing.Any, *args,  **kwargs) -> typing.Any:

        if isinstance(request, str) and "://" in request:
            request = uri_split_as_dict(request)

        if request is _not_found_:
            return kwargs.get("default_value", _not_found_)

        elif isinstance(request, list):
            res = [self._op_fetch(req, *args, **kwargs) for req in request]

        elif not isinstance(request, dict):
            res = request

        elif f"@{SPDB_TAG}" not in request:
            res = {k: self._op_fetch(req, *args, **kwargs) for k, req in request.items()}

        else:

            if request.startswith("@"):
                request = uri_split_as_dict(request[1:])
                res = target.child(request.get("path", None)).fetch(
                    *args, request=request, **kwargs)
            else:
                res = request

            res = target.fetch(*args, request=request,  **kwargs)

        return res
