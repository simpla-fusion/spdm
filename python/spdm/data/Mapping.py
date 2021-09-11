import collections
import os
import pathlib

from ..common.SpObject import SpObject

from ..util.logger import logger
from ..util.PathTraverser import PathTraverser
from ..util.urilib import urisplit
from .Collection import Collection
from .Document import Document
from .Entry import Entry, EntryCombiner
from .File import File
from ..util.utilities import _undefined_

SPDB_XML_NAMESPACE = "{http://fusionyun.org/schema/}"


class MappingEntry(Entry):
    def __init__(self, *args, mapping: Entry = None, source: Entry = None,  ** kwargs):
        super().__init__(*args, **kwargs)
        self._mapping = mapping  # self._data._mapping.entry
        self._source = source

    def __post_process__(self, request, *args, **kwargs):
        if isinstance(request, Entry):
            res = MappingEntry(source=self._source, mapping=request)
        elif isinstance(request, collections.abc.Mapping) and len(request) == 1:
            k, v = next(iter(request.items()))
            if k[0] == "{":
                res = self._source.fetch(k, v)
            else:
                res = super().__post_process__(request, *args, **kwargs)
        else:
            res = super().__post_process__(request, *args, **kwargs)
        return res

    def child(self, path, *args, **kwargs):
        return MappingEntry(source=self._source, mapping=self._mapping.child(path, *args, **kwargs))

    def get(self,  path, *args,  is_raw_path=False,  **kwargs):
        return self.__post_process__(self._mapping.get(path, *args, only_one=True, **kwargs))

    def get_value(self,  path, *args,  is_raw_path=False,  **kwargs):
        return self.__post_process__(self._mapping.get_value(path, *args, **kwargs))

    def put(self,  path, value, *args, is_raw_path=False,   **kwargs):
        if not is_raw_path:
            return PathTraverser(path).apply(lambda p: self.get(p, is_raw_path=True, **kwargs))
        else:
            request = self._mapping.get_value(path, *args, **kwargs)

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


class Mapping(SpObject):
    DEFAULT_GLOBAL_SCHEMA = "imas/3"

    def __init__(self, *args, mapping_path="",   **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(mapping_path, str):
            mapping_path = mapping_path.split(":")
        self._mapping_path = mapping_path + \
            os.environ.get("SP_DATA_MAPPING_PATH", "").split(":")

    def map(self, source: Entry, source_schema=None, target_schema=_undefined_):
        return MappingEntry(source, mapping=self.find(source_schema, target_schema))

    def find(self,  source_schema: str = None, target_schema: str = _undefined_) -> Entry:

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
