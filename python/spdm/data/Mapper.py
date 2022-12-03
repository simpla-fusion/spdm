import collections
import collections.abc
import os
import pathlib

from ..common.PathTraverser import PathTraverser
from ..common.tags import _undefined_
from ..util.logger import logger
from .Document import Document
from .Entry import Entry
from .File import File
from .SpObject import SpObject

SPDB_XML_NAMESPACE = "{http://fusionyun.org/schema/}"
SPDB_TAG = "spdb"


class MapperEntry(Entry):
    def __init__(self, source: Entry, *args, mapping: Entry = None,  ** kwargs):
        super().__init__(*args, **kwargs)
        self._mapping = mapping  # self._data._mapping.entry
        self._source = source

    def __post_process__(self, value, *args, lazy=True, **kwargs):
        if isinstance(value, Entry):
            if value.attribute.get(SPDB_TAG, None) is not None:
                res = self._source.get(value.pull(lazy=False))
            elif lazy:
                res = MapperEntry(self._source, mapping=value)
            else:
                res = self.__post_process__(value.pull(lazy=False), *args, lazy=False, ** kwargs)
        elif isinstance(value, collections.abc.Mapping):
            if f"@{SPDB_TAG}" in value:
                res = self._source.get(value)
            else:
                res = {k: self.__post_process__(v, lazy=lazy, **kwargs) for k, v in value.items()}
            # k, v = next(iter(value.items()))
            # if k[0] == "{":
            #     res = self._source.get(v)
            # else:
            #     logger.warning("INCOMPLETE IMPLEMENDENT!")
            #     res = {k: self.get(v, lazy=lazy, **kwargs) for k, v in value.items()}
        elif isinstance(value, list):
            res = [self.__post_process__(v, *args, lazy=lazy, **kwargs) for v in value]
        # elif getattr(value, "tag", None) is not None:
        #     res = self._source.get(value)
            # raise TypeError(f"[{type(request)}]{request}")
            # res = super().__post_process__(request, *args, **kwargs)
        else:
            res = value

        return res

    def child(self, path, *args, **kwargs):
        return MapperEntry(self._source, mapping=self._mapping.child(path, *args, **kwargs))

    def get(self,  path, *args,  is_raw_path=False,  **kwargs):
        return self.__post_process__(self._mapping.get(path, *args, **kwargs))

    def put(self,  path, value, *args, is_raw_path=False,   **kwargs):
        if not is_raw_path:
            return PathTraverser(path).apply(lambda p: self.get(p, is_raw_path=True, **kwargs))
        else:
            request = self._mapping.pull(path, *args, **kwargs)

            if isinstance(request, str):
                self._data.update(request, value, is_raw_path=True)
            elif isinstance(request, collections.abc.Sequence):
                raise NotImplementedError()
            elif isinstance(request, collections.abc.Mapping):
                raise NotImplementedError()
            elif request is not None:
                raise RuntimeError(f"Can not write to non-empty entry! {path}")

    def iter(self,  request, *args, **kwargs):
        for source_req in self._mapping.iter(request, *args, **kwargs):
            yield self.__post_process__(source_req)

    def pull(self, *args, **kwargs):
        return self.__post_process__(self._mapping.pull(), *args, lazy=False, **kwargs)

    def push(self, value, *args, **kwargs):
        return self.put(None, value, *args, **kwargs)

    def __serialize__(self):
        return self.__post_process__(self._mapping.__serialize__())


class Mapper(SpObject):

    def __init__(self, mapping=[],
                 source_schema=_undefined_,
                 target_schema=_undefined_,
                 envs=None):
        super().__init__()
        if isinstance(mapping, str):
            mapping = mapping.split(":")
        elif isinstance(mapping, pathlib.Path):
            mapping = [mapping]
        elif mapping in (None, _undefined_):
            mapping = []

        mapping += os.environ.get("SP_DATA_MAPPING_PATH", "").split(":")

        self._mapping_path = [pathlib.Path(p) for p in mapping if p not in ('', "")]

        if len(self._mapping_path) == 0:
            raise RuntimeError(f"No mapping file!")

        self._default_source_schema = source_schema if source_schema is not _undefined_ else "EAST"
        self._default_target_schema = target_schema if target_schema is not _undefined_ else "imas/3"
        self._envs = envs

    @property
    def source_schema(self) -> str:
        return self._default_source_schema

    @property
    def target_schema(self) -> str:
        return self._default_target_schema

    def map(self, source: Entry, *args, **kwargs) -> MapperEntry:
        mapping = self.find_mapping(*args, **kwargs)
        if isinstance(mapping, Entry):
            return MapperEntry(source, mapping=mapping)
        elif mapping is None:
            return source
        else:
            raise FileNotFoundError(f"Can not find mapping file!")

    def find_mapping(self,  source_schema: str = _undefined_, target_schema: str = _undefined_) -> Entry:
        if source_schema is _undefined_:
            source_schema = self._default_source_schema
        if target_schema is _undefined_:
            target_schema = self._default_target_schema

        if source_schema == target_schema:
            return None

        map_tag = f"{source_schema}/{target_schema}"

        file_path_suffix = ["config.xml",
                            "static/config.xml", "dynamic/config.xml"]

        mapping_files = []
        for m_dir in self._mapping_path:
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

        return File(mapping_files, mode="r", format="XML").read()


def create_mapper(*args,  source_schema=_undefined_, target_schema=_undefined_, **kwargs) -> Mapper:
    if len(args) > 0 and isinstance(args[0], Mapper):
        return args[0]
    else:
        return Mapper(*args, source_schema=source_schema, target_schema=target_schema, **kwargs)
