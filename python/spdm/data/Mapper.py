from __future__ import annotations

import collections
import collections.abc
import os
import pathlib
import typing

from ..common.PathTraverser import PathTraverser
from ..common.tags import _undefined_
from .Entry import Entry
from .File import File
from .Path import Path
from .SpObject import SpObject

SPDB_XML_NAMESPACE = "{http://fusionyun.org/schema/}"
SPDB_TAG = "spdb"


class PathMapper(Path):
    def __init__(self, mapper: Entry, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mapper = mapper

    def duplicate(self) -> PathMapper:
        other: PathMapper = super().duplicate()
        other._mapper = self._mapper
        return other

    def as_request(self) -> dict:
        return self._mapper.get(self)


class EntryMapper(Entry):

    # def child(self, path, *args, **kwargs):
    #     return MapperEntry(self._source, mapping=self._mapping.child(path, *args, **kwargs))
    # def get(self,  path, *args,  is_raw_path=False,  **kwargs):
    #     return self.__post_process__(self._mapping.get(path, *args, **kwargs))

    # def put(self,  path, value, *args, is_raw_path=False,   **kwargs):
    #     if not is_raw_path:
    #         return PathTraverser(path).apply(lambda p: self.get(p, is_raw_path=True, **kwargs))
    #     else:
    #         request = self._mapping.pull(path, *args, **kwargs)

    #         if isinstance(request, str):
    #             self._data.update(request, value, is_raw_path=True)
    #         elif isinstance(request, collections.abc.Sequence):
    #             raise NotImplementedError()
    #         elif isinstance(request, collections.abc.Mapping):
    #             raise NotImplementedError()
    #         elif request is not None:
    #    raise RuntimeError(f"Can not write to non-empty entry! {path}")

    def iter(self,  request, *args, **kwargs):
        for source_req in self._path.iter(request, *args, **kwargs):
            yield self.__post_process__(source_req)

    def __post_process__(self, value, *args, lazy=True, **kwargs):
        if isinstance(value, Entry):
            if value.attribute.get(SPDB_TAG, None) is not None:
                res = self._source.get(value.pull(lazy=False))
            elif lazy:
                res = EntryMapper(self._source, mapping=value)
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

    def insert(self, value, *args, is_raw_path=False, **kwargs):
        path = self._path
        if not is_raw_path:
            return PathTraverser(path).apply(lambda p: self.get(p, is_raw_path=True, **kwargs))
        else:
            request = self._path.as_request()

            if isinstance(request, str):
                self._cache.update(request, value, is_raw_path=True)
            elif isinstance(request, collections.abc.Sequence):
                raise NotImplementedError()
            elif isinstance(request, collections.abc.Mapping):
                raise NotImplementedError()
            elif request is not None:
                raise RuntimeError(f"Can not write to non-empty entry! {path}")

    def __serialize__(self):
        return self.__post_process__(self._mapping.__serialize__())


class Mapper(SpObject):

    def __init__(self, mapping=[],
                 source_schema: typing.Optional[str] = None,
                 target_schema: typing.Optional[str] = None,
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

        self._default_source_schema: str = source_schema if source_schema is not None else "EAST"
        self._default_target_schema: str = target_schema if target_schema is not None else "imas/3"
        self._envs = envs

    @ property
    def source_schema(self) -> str:
        return self._default_source_schema

    @ property
    def target_schema(self) -> str:
        return self._default_target_schema

    def map(self, source: Entry, *args, **kwargs) -> Entry:
        mapping = self.find_mapping(*args, **kwargs)
        if isinstance(mapping, Entry):
            return EntryMapper(source, mapping=mapping)
        elif mapping is None:
            return source
        else:
            raise FileNotFoundError(f"Can not find mapping file!")

    def find_mapping(self,  source_schema: typing.Optional[str] = None, target_schema: typing.Optional[str] = None) -> typing.Optional[Entry]:
        if source_schema is None:
            source_schema = self._default_source_schema
        if target_schema is None:
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


def create_mapper(*args,  source_schema: typing.Optional[str] = None, target_schema: typing.Optional[str] = None, **kwargs) -> Mapper:
    if len(args) > 0 and isinstance(args[0], Mapper):
        return args[0]
    else:
        return Mapper(*args, source_schema=source_schema, target_schema=target_schema, **kwargs)
