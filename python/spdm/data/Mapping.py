import collections
import os
import pathlib

from ..util.dict_util import format_string_recursive
from ..util.LazyProxy import LazyProxy
from ..util.logger import logger
from ..util.PathTraverser import PathTraverser
from ..util.urilib import urisplit
from .Collection import Collection
from .Document import Document
from .Entry import Entry
from .File import File

SPDB_XML_NAMESPACE = "{http://fusionyun.org/schema/}"


class MappingEntry(Entry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mapping = self._holder._mapping.entry

    def get(self,  path, *args,  is_raw_path=False,  **kwargs):
        # if not is_raw_path:
        #     res = PathTraverser(path).apply(lambda p, _s=self: _s.get(p,  is_raw_path=True, **kwargs))
        # else:
        return self._holder.fetch(self._mapping.get_value(path, *args, **kwargs))

    def put(self,  path, value, *args, is_raw_path=False,   **kwargs):
        if not is_raw_path:
            return PathTraverser(path).apply(lambda p: self.get(p, is_raw_path=True, **kwargs))
        else:
            request = self._mapping.get_value(path, *args, **kwargs)

            if isinstance(request, str):
                self._holder.update(request, value, is_raw_path=True)
            elif isinstance(request, collections.abc.Sequence):
                raise NotImplementedError()
            elif isinstance(request, collections.abc.Mapping):
                raise NotImplementedError()
            elif request is not None:
                raise RuntimeError(f"Can not write to non-empty entry! {path}")

    def iter(self,  request, *args, **kwargs):
        for source_req in self._mapping.iter(request, *args, **kwargs):
            yield self._holder.fetch(source_req)


class MappingDocument(Document):
    def __init__(self, *args,  source=None,  mapping=None,  **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(mapping, Document):
            self._mapping = Document(mapping)
        else:
            self._mapping = mapping

        self._source = source or {}

    @property
    def entry(self):
        return MappingEntry(self)

    def _wrapper(self, tag):
        doc = self._source.get(tag, None)
        if not doc:
            doc = self._parent.source_wrapper(tag).find_one(fid=self.fid)
            self._source[tag] = doc

        return doc

    def fetch(self, request, tag=None):
        res = None
        if isinstance(tag, str) and "}" in tag:
            handler = self._wrapper(tag)
            res = handler.fetch(format_string_recursive(request, self._envs))
        elif isinstance(request,  collections.abc.Sequence) and not isinstance(request, str):
            res = [self.fetch(v) for v in request]
        elif isinstance(request, collections.abc.Mapping):
            res = {k: self.fetch(v, tag=k) for k, v in request.items()}
        else:
            res = request
        return res

    def update(self, request, value=None, tag=None):
        pass


class MappingCollection(Collection):

    def __init__(self,  metadata=None, *args, source=None, mapping=None, id_hasher=None, ** kwargs):
        super().__init__(metadata, *args, schema="mapping", **kwargs)

        id_hasher = id_hasher or "{shot}"

        if callable(id_hasher):
            self._id_hasher = id_hasher
        else:
            self._id_hasher = lambda *a, _pattern=id_hasher, **k: _pattern.format(**k)

        mapping_schema = "EAST"
        mapping_version = "imas/3"

        if isinstance(metadata, str):
            npos = metadata.find("+")
            if not npos:
                raise ValueError(metadata)

            mapping_schema = mapping_schema or metadata[:npos]
            source = source or metadata[npos+1:]

        if isinstance(mapping, Document):
            self._mapping = mapping
        else:
            if isinstance(mapping, str):
                mapping_path = mapping.split(";")
            elif isinstance(mapping, pathlib.PosixPath):
                mapping_path = [mapping]
            elif isinstance(mapping, collections.abc.Sequence):
                mapping_path = []
            elif isinstance(mapping, collections.abc.Mapping):
                dpth = mapping.get("path", [])
                if isinstance(dpth, str):
                    dpth = dpth.split(";")
                mapping_path = dpth
                mapping_version = mapping.get("version", mapping_version)
                mapping_schema = mapping.get("schema", mapping_schema)

            mapping_path.extend(os.environ.get("SP_DATA_MAPPING_PATH", "").split(";"))

            file_path_suffix = ["config.xml", "static/config.xml", "dynamic/config.xml"]

            mapping_conf_files = []
            for m_dir in mapping_path:
                if not m_dir:
                    continue
                elif isinstance(m_dir, str):
                    m_dir = pathlib.PosixPath(m_dir)
                for file_name in file_path_suffix:
                    p = m_dir / ('/'.join([mapping_schema, mapping_version, file_name]))
                    if p.exists():
                        mapping_conf_files.append(p)

            self._mapping = File(mapping_conf_files, format="XML")

        self._source = {}

        tag = "local"
        if isinstance(source, str):
            source = urisplit(source)
            self._source[source.get("schema", tag)] = source
        elif isinstance(source, collections.abc.Mapping) and "schema" not in source:
            self._source = source
        elif isinstance(source, Collection):
            self._source[source.schema] = source
        else:
            logger.error(f"Illegal taget {source}")

    def __del__(self):
        self._source = {}
        self._mapping = None
        super().__del__()

    def source_wrapper(self, tag):
        handler = self._source.get(tag, None) or self._source.get(tag[tag.rfind("}")+1:], None)

        if isinstance(handler, Collection):
            pass
        elif handler is not None:
            handler = Collection(handler)
            self._source[tag] = handler
        else:
            raise KeyError(tag)

        return handler

    def guess_id(self,   **kwargs):
        return self._id_hasher(**kwargs)

    def insert_one(self, *args, **kwargs):
        return MappingDocument(*args, fid=self.guess_id(**kwargs) or self.next_id,  parent=self,
                               mapping=self._mapping, mode="x", **kwargs)

    def find_one(self,   *args,   **kwargs):
        return MappingDocument(*args, fid=self.guess_id(**kwargs), parent=self,
                               mapping=self._mapping,   mode=self._mode, **kwargs)


__SP_EXPORT__ = MappingCollection
