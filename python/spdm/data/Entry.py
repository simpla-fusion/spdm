from __future__ import annotations
import numpy as np
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

PROTOCOL_LIST = ["local", "file", "http", "https", "ssh", "mdsplus"]


class Entry(Pluggable):
    _plugin_prefix = "spdm.plugins.data.plugin_"
    _plugin_registry = {}

    def __init__(self, data: typing.Any = None, path: Path | PathLike = None, *args, scheme=None, **kwargs,):
        if self.__class__ is not Entry:
            pass
        elif scheme is not None or isinstance(data, (str, URITuple, pathlib.Path)):
            if scheme is None:
                scheme = uri_split(data).protocol

            if isinstance(scheme, str) and scheme != "":
                super().__dispatch_init__([scheme, EntryProxy], self, data, *args, **kwargs)
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

    def __str__(self) -> str: return f'<{self.__class__.__name__} path="{self._path}" />'

    def __getitem__(self, *args) -> Entry: return self.child(*args)

    def __setitem__(self, path, value): return self.child(path).update(value)

    def __delitem__(self, *args): return self.child(*args).remove()

    @property
    def __entry__(self) -> Entry: return self

    @property
    def is_writable(self) -> bool: return False

    @property
    def path(self) -> Path: return self._path

    @property
    def is_leaf(self) -> bool: return self.fetch(Path.tags.is_leaf)

    @property
    def is_list(self) -> bool: return self.fetch(Path.tags.is_list)

    @property
    def is_dict(self) -> bool: return self.fetch(Path.tags.is_dict)

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

        other = self.__copy__()
        other._path.append(path)
        return other

    def next(self, inc: int = 1) -> Entry:
        if not isinstance(self._path[-1], int) and not np.issubdtype(type(self._path[-1]), np.integer):
            raise RuntimeError(f"Path must be end with int! {self._path[-1]} {type(self._path[-1])}")

        next_ = self.__copy__()

        next_._path[-1] += inc

        return next_

    ###########################################################

    @property
    def __value__(self) -> typing.Any:
        return (self._data if len(self._path) == 0 else self.get(default_value=_not_found_))

    def get(self, query=None, default_value: typing.Any = _undefined_, **kwargs) -> typing.Any:
        if query is None:
            entry = self
            args = ()
        elif isinstance(query, (slice, set, dict)):
            entry = self
            args = (query,)
        else:
            entry = self.child(query)
            args = ()

        res = entry.fetch(Path.tags.fetch, *args, default_value=default_value, **kwargs)

        if res is _undefined_:
            raise RuntimeError(f'Can not find "{query}" in {self}')
        else:
            return res

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

    def insert(self, value, **kwargs) -> Entry:
        self._data, next_path = self._path.insert(self._data, value, **kwargs)
        return self.__class__(self._data, next_path)

    def update(self, value, **kwargs) -> Entry:
        self._data = self._path.update(self._data, value, **kwargs)
        return self

    def remove(self, **kwargs) -> int:
        self._data, num = self._path.remove(self._data, **kwargs)
        return num

    def fetch(self, op=None, *args, **kwargs) -> typing.Any:
        """
        Query the Entry.
        Same function as `find`, but put result into a contianer.
        Could be overridden by subclasses.
        """
        return self._path.fetch(self._data, op, *args, **kwargs)

    def for_each(self, *args, **kwargs) -> typing.Generator[typing.Tuple[int, typing.Any], None, None]:
        """Return a generator of the results."""
        yield from self._path.for_each(self._data, *args, **kwargs)

    @deprecated
    def find_next(self, start: int | None, *args, **kwargs) -> typing.Tuple[typing.Any, int | None]:
        """
        Find the value from the cache.
        Return a generator of the results.
        Could be overridden by subclasses.
        """
        return self._path.find_next(self._data, start, *args, **kwargs)

    def find(self, *args, **kwargs) -> Entry:
        raise NotImplementedError()
    ###########################################################


class ChainEntry(Entry):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._entrys: typing.List[Entry] = list(args)
        for idx, v in enumerate(self._entrys):
            if not isinstance(v, Entry):
                self._entrys[idx] = _open_entry(v, **kwargs)

    def __copy_from__(self, other: Entry) -> ChainEntry:
        self._data = other._data
        self._path = copy(other._path)
        self._entrys = getattr(other, "_entrys", [])
        return self

    @property
    def is_writable(self) -> bool: return self._entrys[0].is_writable

    def fetch(self, *args, default_value=_not_found_, **kwargs):
        res = super().fetch(*args, default_value=_not_found_, **kwargs)

        if res is _not_found_:
            for e in self._entrys:
                res = e.child(self._path).fetch(*args, default_value=_not_found_, **kwargs)
                if res is not _not_found_:
                    break

        if res is _not_found_:
            res = default_value

        return res

    def for_each(self, *args, **kwargs) -> typing.Generator[typing.Tuple[int, typing.Any], None, None]:
        for idx, e in self._entrys[0].child(self._path).for_each():
            yield idx, ChainEntry(e, *[o.child(self._path[:] + [idx]) for o in self._entrys[1:]])

    def find(self, *args, **kwargs):
        return ChainEntry(*[e.find(*args, **kwargs) for e in self._entrys])

    @property
    def exists(self) -> bool:
        res = [super().fetch(Path.tags.exists)]
        res.extend([e.child(self._path).fetch(Path.tags.exists) for e in self._entrys])
        return any(res)


def _open_entry(url: str | URITuple | pathlib.Path | Entry, **kwargs) -> Entry:
    """
    Open an Entry from a URL.

    Using urllib.urlparse to parse the URL.  rfc3986

    URL format: <protocol>://<authority>/<path>?<query>#<fragment>

    RF3986 = r"^((?P<protocol>[^:/?#]+):)?(//(?P<authority>[^/?#]*))?(?P<path>[^?#]*)(\\?(?P<query>[^#]*))?(#(?P<fragment>.*))?")

    Example:
        ../path/to/file.json                    => File
        file:///path/to/file                    => File
        ssh://a.b.c.net/path/to/file            => ???
        https://a.b.c.net/path/to/file          => ???

        imas+ssh://a.b.c.net/path/to/file

        east+mdsplus://<mds_prefix>
        east+mdsplus+ssh://<mds_prefix>

    """

    if isinstance(url, Entry) and len(kwargs) == 0:
        return url

    from .File import File

    if isinstance(url, str) and "." not in url and "/" not in url:
        url = f"{url}+://"

    url_ = uri_split(url)

    if not isinstance(url_.path, str):
        raise RuntimeError(f"")

    fragment = url_.fragment

    query = merge_tree_recursive(url_.query, kwargs)

    uid = query.pop("uid", None)

    if isinstance(uid, str):
        shot, *run = uid.split("_")
        run = "_".join(run)
    else:
        shot = query.pop("shot", None) or ""
        run = query.pop("run", None)
        if run is None:
            uid = shot
        else:
            uid = f"{run}_{shot}"

    query["uid"] = uid
    query["shot"] = shot
    query["run"] = run

    global_schema = query.pop("global_schema", None)

    local_schema = query.pop("local_schema", None) or query.pop("device", None)

    schemas = [s for s in url_.protocol.split("+") if s != ""]

    if len(schemas) > 0 and schemas[0] not in PROTOCOL_LIST:
        local_schema = schemas[0]
        schemas = schemas[1:]

    new_url = URITuple(
        protocol="+".join(schemas),
        authority=url_.authority,
        path=url_.path,
        query=query,
        fragment="",
    )

    if local_schema is not None and global_schema != local_schema:
        entry = EntryProxy(
            new_url, local_schema=local_schema, global_schema=global_schema, **query
        )

    elif new_url.protocol.startswith(("local+", "file+")) or (
        new_url.protocol == "" and new_url.path != ""
    ):
        entry = File(new_url, **query).read()

    elif new_url.protocol.startswith(("http", "https", "ssh")):
        raise NotImplementedError(f"{new_url}")

    else:
        entry = Entry(url_, **query)

    if fragment:
        entry = entry.child(fragment.replace(".", "/"))

    return entry


def open_entry(entry, **kwargs) -> Entry:
    if not isinstance(entry, list):
        entry = [entry]

    entry = [a for a in entry if a is not None and a is not _not_found_]

    if len(entry) == 0:
        return None

    elif len(entry) > 1:
        return ChainEntry(*entry, **kwargs)

    else:
        return _open_entry(entry[0], **kwargs)

    # url = uri_split(url_s)

    # # scheme = url.protocol.split("+")

    # # local_schema = None

    # # global_schema = schema

    # # match len(scheme):
    # #     case 0:
    # #         url.protocol = 'local'
    # #     case 1:
    # #         if scheme[0] not in _predefined_protocols:
    # #             local_schema = scheme[0]
    # #             url.protocol = ""
    # #     case _:
    # #         if scheme[0] in _predefined_protocols:
    # #             url.protocol = scheme[0]
    # #             if scheme[1] != kwargs.setdefault("format", scheme[1]):
    # #                 raise ValueError(f"Format mismatch! {scheme[1]} != {kwargs['format']}")
    # #         else:
    # #             local_schema = scheme[0]
    # #             url.protocol = "+".join(scheme[1:])

    # # if local_schema is not None:
    # #     kwargs.update(url.query)
    # #     if url.authority != '' or url.path != '':
    # #         kwargs['netloc'] = url.authority
    # #         kwargs['path'] = url.path
    # #     return EntryProxy(local_schema=local_schema, global_schema=global_schema, ** kwargs)

    # if url.protocol in ["file", "local", "", None]:
    #     from .File import File
    #     return File(url, *args, **kwargs).read()

    # elif url.protocol in ["https", "http"]:
    #     return Entry(url, *args, **kwargs)

    # else:
    #     raise RuntimeError(f"Unknown url {url} {Entry._plugin_registry}")


def asentry(obj, *args, **kwargs) -> Entry:
    if isinstance(obj, Entry):
        entry = obj
    elif isinstance(obj, (str, URITuple, pathlib.Path)):
        entry = open_entry(obj, *args, **kwargs)
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

    if hasattr(obj, "entry"):
        obj = obj.entry
    if obj is None:
        obj = default_value

    if obj is None or not dataclasses.is_dataclass(dclass) or isinstance(obj, dclass):
        pass
    # elif getattr(obj, 'empty', False):
    #   obj = None
    elif dclass is array_type:
        obj = as_array(obj)
    elif hasattr(obj.__class__, "get"):
        obj = dclass(
            **{
                f.name: as_dataclass(
                    f.type,
                    obj.get(
                        f.name,
                        f.default if f.default is not dataclasses.MISSING else None,
                    ),
                )
                for f in dataclasses.fields(dclass)
            }
        )
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
        return sum(
            [first, *(v for v in others if (v is not None and v is not _not_found_))]
        )
    elif len(others) > 1:
        return deep_reduce(first, deep_reduce(others, level=level), level=level)
    elif others[0] is None or first is _not_found_:
        return first
    elif isinstance(first, collections.abc.Sequence):
        if isinstance(others[0], collections.abc.Sequence) and not isinstance(
            others, str
        ):
            return [*first, *others[0]]
        else:
            return [*first, others[0]]
    elif isinstance(first, collections.abc.Mapping) and isinstance(
        others[0], collections.abc.Mapping
    ):
        second = others[0]
        res = {}
        for k, v in first.items():
            res[k] = deep_reduce(v, second.get(k, None), level=level - 1)
        for k, v in second.items():
            if k not in res:
                res[k] = v
        return res
    elif others[0] is None or others[0] is _not_found_:
        return first
    else:
        raise TypeError(f"Can not merge dict with {others}!")


def convert_fromentry(cls, obj, *args, **kwargs):
    origin_type = getattr(cls, "__origin__", cls)
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
    _maps = {}
    _mapping_path = []
    _default_local_schema: str = "EAST"
    _default_global_schema: str = "imas/3"

    @classmethod
    def load(cls, url: str | None = None, local_schema: str = None, global_schema: str = None, **kwargs):
        """检索并导入 mapping files

        mapping files 目录结构约定为 :

        - <local schema>/<global schema>
            - config.xml
            - static            # 存储静态数据，例如装置描述文件
                - config.xml
                - <...>

            - protocol0         # 存储 protocol0 所对应mapping，例如 mdsplus
                - config.xml
                - <...>

            - protocol1         # 存储 protocol1 所对应mapping，例如 hdf5
                - config.xml
                - <...>

        Example:
          1. east+mdsplus://.... 对应的目录结构为
            - east/imas/3
                - static
                    - config.xml
                    - wall.xml
                    - pf_active.xml  (包含 pf 线圈几何信息)
                    - ...
                - mdsplus
                    - config.xml (包含<spdb > 描述子数据库entry )
                    - pf_active.xml


        """
        from .File import File

        mapper_list = EntryProxy._maps

        _url = uri_split(url)

        kwargs = merge_tree_recursive(url.query, kwargs)

        enabled_entry = kwargs.pop("enable", "").split(",")

        if local_schema is None:
            local_schema = EntryProxy._default_local_schema

        if global_schema is None:
            global_schema = EntryProxy._default_global_schema

        map_tag = [local_schema.lower(), global_schema.lower()]

        if _url.protocol != "":
            map_tag.append(_url.protocol)

        map_tag_str = "/".join(map_tag)

        mapper = mapper_list.get(map_tag_str, _not_found_)

        if mapper is _not_found_:
            prefix = "/".join(map_tag[:2])

            config_files = [
                f"{prefix}/config.xml",
                f"{prefix}/static/config.xml",
                f"{prefix}/{local_schema.lower()}.xml",
            ]

            if len(map_tag) > 2:
                config_files.append(f"{'/'.join(map_tag[:3])}/config.xml")

            mapping_files: typing.List[pathlib.Path] = []

            for m_dir in EntryProxy._mapping_path:
                if not m_dir:
                    continue
                elif isinstance(m_dir, str):
                    m_dir = pathlib.Path(m_dir)

                for file_name in config_files:
                    p = m_dir / file_name
                    if p.exists():
                        mapping_files.append(p)

            if len(mapping_files) == 0:
                raise FileNotFoundError(
                    f"Can not find mapping files for {map_tag} MAPPING_PATH={EntryProxy._mapping_path} !"
                )

            mapper = File(mapping_files, mode="r", format="XML").read()

            mapper_list[map_tag_str] = mapper

        entry_list = {}

        spdb = mapper.child("spdb").fetch()

        if not isinstance(spdb, dict):
            entry_list["*"] = _url
        else:
            attr = {k[1:]: v for k, v in spdb.items() if k.startswith("@")}

            attr["prefix"] = f"{_url.protocol}://{_url.authority}{_url.path}"

            attr.update(kwargs)

            for entry in spdb.get("entry", []):
                id = entry.get("@id", None)

                enable = entry.get("@enable", "true") == "true"

                if id is None:
                    continue
                elif not enable and id not in enabled_entry:
                    continue

                entry_list[id] = entry.get("_text", "").format(**attr)

        return mapper, entry_list

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 2 and isinstance(args[0], Entry) and isinstance(args[1], dict):
            self._mapper = args[0]
            self._entry_list = args[1]
        else:
            self._mapper, self._entry_list = self.__class__.load(*args, **kwargs)

    def __copy__(self) -> Entry:
        obj = object.__new__(self.__class__)
        obj.__copy_from__(self)
        obj._mapper = self._mapper
        obj._entry_list = self._entry_list
        return obj

    def child(self, *args, **kwargs) -> Entry:
        res = super().child(*args, **kwargs)
        res._entry_list = self._entry_list
        return res

    def insert(self, value, **kwargs) -> Entry:
        raise NotImplementedError(f"")

    def update(self, value, **kwargs) -> Entry:
        raise NotImplementedError(f"")

    def remove(self, **kwargs) -> int:
        raise NotImplementedError(f"")

    def fetch(self, *args, default_value=_not_found_, **kwargs) -> typing.Any:
        request = self._mapper.child(self._path).fetch(*args, default_value=_not_found_, lazy=False, **kwargs)

        return self._op_fetch(request, default_value=default_value)

    def for_each(self, *args, **kwargs) -> typing.Generator[typing.Tuple[int, typing.Any], None, None]:
        """Return a generator of the results."""
        for idx, request in self._mapper.child(self._path).for_each(*args, **kwargs):
            yield idx, self._op_fetch(request)

    def find(self, *args, **kwargs):
        raise NotImplementedError()

    def _get_entry_by_name(self, entry_name: str, default_value=None) -> Entry | None:
        entry = self._entry_list.get(entry_name, None)

        if isinstance(entry, (str, URITuple)):
            entry = open_entry(entry)
            self._entry_list[entry_name] = entry

        if isinstance(entry, Entry):
            pass
        elif default_value is _not_found_:
            raise RuntimeError(f"Can not find entry for {entry_name}")
        else:
            entry = default_value

        return entry

    def _op_fetch(self, request: typing.Any, *args, **kwargs) -> typing.Any:
        if isinstance(request, str) and "://" in request:
            request = uri_split_as_dict(request)

        if request is _not_found_ or request is None:
            defaultentry = self._get_entry_by_name("*", None)
            if defaultentry is None:
                res = _not_found_
            else:
                res = defaultentry.child(self._path).fetch(None, *args, **kwargs)

        elif isinstance(request, Entry):
            res = EntryProxy(request, self._entry_list)

        elif isinstance(request, list):
            res = [self._op_fetch(req, *args, **kwargs) for req in request]

        elif not isinstance(request, dict):
            res = request

        elif "@spdb" not in request:
            res = {k: self._op_fetch(req, *args, **kwargs) for k, req in request.items()}

        else:
            entry = self._get_entry_by_name(request.get("@spdb", None))

            if not isinstance(entry, Entry):
                raise RuntimeError(f"Can not find entry for {request}")

            res = entry.fetch(request.get("_text"), *args, **kwargs)

        return res


EntryProxy._mapping_path.extend([pathlib.Path(p)
                                for p in os.environ.get("SP_DATA_MAPPING_PATH", "").split(":") if p != ""])
