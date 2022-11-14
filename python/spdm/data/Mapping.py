import collections
import collections.abc
from linecache import lazycache
import os
import pathlib

from spdm.logger import logger
from spdm.plugins.data.file.PluginXML import XMLEntry
from spdm.SpObject import SpObject
from spdm.tags import _undefined_

from ..util.PathTraverser import PathTraverser
from .Document import Document
from .Entry import Entry
from .File import File

SPDB_XML_NAMESPACE = "{http://fusionyun.org/schema/}"
SPDB_TAG = "spdb"


class MappingEntry(Entry):
    def __init__(self, *args, mapping: Entry = None, source: Entry = None,  ** kwargs):
        super().__init__(*args, **kwargs)
        self._mapping = mapping  # self._data._mapping.entry
        self._source = source

    def __post_process__(self, value, *args, lazy=True, **kwargs):
        if isinstance(value, Entry):
            if value.attribute.get(SPDB_TAG, None) is not None:
                res = self._source.get(value.pull(lazy=False))
            elif lazy:
                res = MappingEntry(source=self._source, mapping=value)
            else:
                res = self.__post_process__(value.pull(lazy=False), *args, lazy=False, ** kwargs)
        elif isinstance(value, collections.abc.Mapping) and len(value) == 1:
            k, v = next(iter(value.items()))
            if k[0] == "{":
                res = self._source.get(v)
            else:
                logger.warning("INCOMPLETE IMPLEMENDENT!")
                res = {k: self.get(v, lazy=lazy, **kwargs) for k, v in value.items()}
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
        return MappingEntry(source=self._source, mapping=self._mapping.child(path, *args, **kwargs))

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

    def pull(self):
        return self.__post_process__(self._mapping.pull(), lazy=False)

    def push(self, value, *args, **kwargs):
        return self.put(None, value, *args, **kwargs)

    def __serialize__(self):
        return self.__post_process__(self._mapping.__serialize__())


class Mapping(SpObject):
    DEFAULT_GLOBAL_SCHEMA = "imas/3"

    def __init__(self, *args, mapping_path="", global_schema=_undefined_,   **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(mapping_path, str):
            mapping_path = mapping_path.split(":")
        self._mapping_path = mapping_path + \
            os.environ.get("SP_DATA_MAPPING_PATH", "").split(":")
        if global_schema is _undefined_:
            self._global_schema = Mapping.DEFAULT_GLOBAL_SCHEMA

    def map(self, source: Entry, source_schema=None, **kwargs):
        if isinstance(source, Document):
            source_schema = getattr(source, "schema", source_schema)
            source = source.entry
        elif isinstance(source, Entry):
            pass
        elif isinstance(source, collections.abc.Sequence):
            source = Entry.create(source)
        elif isinstance(source, collections.abc.Mapping):
            if "$class" in source:
                source_schema = source.get("schema", None)
                source = Entry.create(source)
            else:
                raise NotImplementedError(f"TODO: Multi-sources mapper ")
        else:
            raise NotImplementedError()

        return MappingEntry(source=source, mapping=self.find(source_schema, **kwargs))

    def find(self,  source_schema: str = None, target_schema: str = _undefined_) -> Entry:
        if target_schema is _undefined_:
            target_schema = self._global_schema

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

        # return EntryCombiner([File(fid, mode="r", format="XML").read() for fid in mapping_conf_files])
        return File(mapping_files, mode="r", format="XML").read()
