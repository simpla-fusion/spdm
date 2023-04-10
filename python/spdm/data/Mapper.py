from __future__ import annotations

import collections
import collections.abc
import os
import pathlib
import typing

from ..common.PathTraverser import PathTraverser
from ..common.tags import _undefined_
from ..util.logger import logger
from ..util.uri_utils import uri_split_as_dict
from .Entry import Entry, as_entry
from .File import File
from .Path import Path
from .SpObject import SpObject

SPDB_XML_NAMESPACE = "{http://fusionyun.org/schema/}"
SPDB_TAG = "spdb"


class MapperPath(Path):
    def __init__(self, *args, mapping: typing.Optional[Entry] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._mapping = as_entry(mapping)

    def duplicate(self, new_value=None) -> MapperPath:
        return MapperPath(self[:] if new_value is None else new_value, mapping=self._mapping)

    def find(self, target: typing.Any, *args, **kwargs) -> typing.Generator[typing.Any, None, None]:
        target = as_entry(target)
        for request in self._mapping.child(self[:]).find(*args, **kwargs):
            yield self._fetch(target, request)

    def query(self, target: typing.Any, *args, **kwargs) -> typing.Any:
        request = self._mapping.child(self[:]).query(*args, **kwargs)
        return self._fetch(as_entry(target), request)

    def insert(self, target: typing.Any, *args, **kwargs) -> int:
        raise NotImplementedError("Not implemented yet!")

    def update(self, target: typing.Any, *args, **kwargs) -> int:
        raise NotImplementedError("Not implemented yet!")

    def remove(self, target: typing.Any, *args, **kwargs) -> int:
        raise NotImplementedError("Not implemented yet!")

    def _fetch(self, target: Entry,  request: typing.Any, *args, **kwargs):
        if isinstance(request, str):
            if request.startswith("@"):
                request = uri_split_as_dict(request[1:])
                res = target.child(request.get("path", None)).query(*args, request=request, **kwargs)
            else:
                res = request
        elif isinstance(request, collections.abc.Sequence):
            res = [self._fetch(target, v, *args,  **kwargs) for v in request]
        elif isinstance(request, collections.abc.Mapping):
            if f"@{SPDB_TAG}" in request:
                res = target.child(request.get("path", None)).query(*args, request=request, **kwargs)
            else:
                res = {k: self._fetch(target, v, *args, **kwargs) for k, v in request.items()}
        else:
            res = request

        return res


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
        if mapping is None:
            return source
        else:
            return Entry(as_entry(source), path=MapperPath(mapping=mapping))

    def find_mapping(self,  source_schema: typing.Optional[str] = None, target_schema: typing.Optional[str] = None) -> typing.Optional[Entry]:
        if source_schema is None:
            source_schema = self._default_source_schema
        if target_schema is None:
            target_schema = self._default_target_schema

        if source_schema == target_schema:
            logger.debug(f"Source and target schema are the same! {source_schema}")
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

        return File.create(mapping_files, mode="r", format="XML").read()


def create_mapper(*args,  source_schema: typing.Optional[str] = None, target_schema: typing.Optional[str] = None, **kwargs) -> Mapper:
    if len(args) > 0 and isinstance(args[0], Mapper):
        return args[0]
    else:
        return Mapper(*args, source_schema=source_schema, target_schema=target_schema, **kwargs)
