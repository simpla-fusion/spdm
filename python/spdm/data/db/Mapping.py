import collections
import pathlib
import os

from spdm.util.urilib import urisplit
from spdm.util.AttributeTree import AttributeTree
from spdm.util.dict_util import format_string_recursive
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.PathTraverser import PathTraverser

from ..Collection import Collection
from ..Document import Document
from ..Entry import Entry
from ..File import File


class MappingEntry(Entry):
    def __init__(self, holder, *args, mapping=None, envs=None, **kwargs):
        super().__init__(holder, *args, **kwargs)
        if isinstance(mapping, Entry):
            self._mapping = mapping
        else:
            raise TypeError(mapping)

        self._envs = envs or {}

    def put(self,  path, value, *args, is_raw_path=False,   **kwargs):
        if not is_raw_path:
            return PathTraverser(path).apply(lambda p: self.get(p, is_raw_path=True, **kwargs))
        else:
            req = self._mapping.handler.get(self._mapping.holder, path, *args, **kwargs)

            if isinstance(req, str):
                req = format_string_recursive(req, self._envs)
                self._target.put(req, value, is_raw_path=True)
            elif isinstance(req, collections.abc.Sequence):
                raise NotImplementedError()
            elif isinstance(req, collections.abc.Mapping):
                raise NotImplementedError()
            elif req is not None:
                raise RuntimeError(f"Can not write to non-empty entry! {path}")

    def _fetch(self,  item, *args, **kwargs):
        res = None
        if item is None:
            pass
        elif isinstance(item, LazyProxy):
            res = MappingEntry(self._holder, mapping=item.__object__).entry
        # elif isinstance(item, Entry):
        #     logger.debug(item.holder.data)
        #     return LazyProxy(holder, handler=MappingEntry(self._target, mapping=Entry(item, handler=self._mapping.handler)))
        elif isinstance(item, list):
            res = [self._fetch(v) for v in item]
        elif not isinstance(item, collections.abc.Mapping):
            res = item
        elif "{http://fusionyun.org/schema/}data" in item:
            if self._holder is None:
                raise RuntimeError()
            req = item["{http://fusionyun.org/schema/}data"]

            req = format_string_recursive(req, self._envs)
            res = self._holder.get(req, *args, **kwargs)
        else:
            res = {k: self._fetch(v) for k, v in item.items()}

        if isinstance(res, collections.abc.Mapping):
            res = AttributeTree(res)

        return res

    def get(self,  path, *args,  is_raw_path=False,  **kwargs):
        if not is_raw_path:
            res = PathTraverser(path).apply(lambda p, _s=self: _s.get(p,  is_raw_path=True, **kwargs))
        else:
            target_path = self._mapping.get_value(path, *args, **kwargs)
            res = self._fetch(target_path)
        return res

    def iter(self,  path, *args, **kwargs):
        for target_path in self._mapping.iter(path, *args, **kwargs):
            yield self._fetch(target_path)


class MappingDocument(Document):
    def __init__(self,  *args, target=None,  mapping=None, envs=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(mapping, Document):
            self._mapping = Document(mapping).root
        else:
            self._mapping = mapping.root
        self._target = target

        self._envs = envs or {}

    # def __del__(self):
    #     del self._target
    #     del self._mapping
    #     super().__del__()

    @property
    def root(self):
        return MappingEntry(self._target.root, mapping=self._mapping, envs=self._envs)


class MappingCollection(Collection):
    def __init__(self, desc, *args, target=None, mapping=None, mapping_data_path=None, mapping_version="imas/3", id_hasher=None, **kwargs):
        super().__init__(desc, *args, id_hasher=id_hasher or "{shot}", **kwargs)

        if isinstance(desc, str):
            npos = desc.find("+")
            if not npos:
                raise ValueError(desc)

            mapping = mapping or desc[:npos]
            target = target or desc[npos+1:]

        if not isinstance(target, Collection):
            self._target = Collection(target)
        else:
            self._target = target

        if isinstance(mapping, Document):
            self._mapping = Document(mapping, envs=self.envs)
            return

        if isinstance(mapping_data_path, str):
            mapping_data_path = mapping_data_path.split(";")
        elif isinstance(mapping_data_path, pathlib.PosixPath):
            mapping_data_path = [mapping_data_path]
        elif mapping_data_path is None:
            mapping_data_path = []

        mapping_data_path.extend(os.environ.get("SP_DATA_MAPPING_PATH", "").split(";"))

        file_path_suffix = ["config.xml", "static/config.xml", "dynamic/config.xml"]

        mapping_conf_files = []
        for m_dir in mapping_data_path:
            if not m_dir:
                continue
            elif isinstance(m_dir, str):
                m_dir = pathlib.PosixPath(m_dir)
            for file_name in file_path_suffix:
                p = m_dir / ('/'.join([mapping, mapping_version, file_name]))
                if p.exists():
                    mapping_conf_files.append(p)

        self._mapping = File(path=mapping_conf_files, file_format="XML")

    # def __del__(self):
    #     del self._target
    #     super().__del__()

    def insert_one(self, *args,  query=None, **kwargs):
        oid = self.guess_id(*args, **collection.ChainMap(query or {}, kwargs))
        doc = self._target.insert_one(oid)
        return MappingDocument(target=doc, envs=collections.ChainMap(kwargs, self._envs), fid=oid, mapping=self._mapping)

    def find_one(self,   *args, query=None,  **kwargs):
        oid = self.guess_id(*args, **collections.ChainMap(query or {}, kwargs))
        doc = self._target.find_one(oid)
        return MappingDocument(target=doc, envs=collections.ChainMap(kwargs, self._envs), fid=oid,   mapping=self._mapping)


__SP_EXPORT__ = MappingCollection
