import collections
import pathlib

from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.PathTraverser import PathTraverser
from spdm.util.AttributeTree import AttributeTree
from ..Collection import Collection
from ..Document import Document
from ..Node import Node


class MappingNode(Node):
    def __init__(self, *args, mapping=None, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(mapping, Node):
            self._mapping = mapping
        else:
            raise TypeError(mapping)

    def put(self,  path, value, *args, is_raw_path=False,   **kwargs):
        if not is_raw_path:
            return PathTraverser(path).apply(lambda p: self.get(p, is_raw_path=True, **kwargs))
        else:
            req = self._mapping.handler.get(self._mapping.holder, path, *args, **kwargs)
            if isinstance(req, str):
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
            res = MappingNode(self._holder, mapping=item.__object__).entry
        # elif isinstance(item, Node):
        #     logger.debug(item.holder.data)
        #     return LazyProxy(holder, handler=MappingNode(self._target, mapping=Node(item, handler=self._mapping.handler)))
        elif isinstance(item, list):
            res = [self._fetch(v) for v in item]
        elif not isinstance(item, collections.abc.Mapping):
            res = item
        elif "{http://hpc.ipp.ac.cn/SpDB}data" in item:
            if self._holder is None:
                raise RuntimeError()
            res = self._holder.get(item["{http://hpc.ipp.ac.cn/SpDB}data"], *args, **kwargs)
        else:
            res = {k: self._fetch(v) for k, v in item.items()}

        if isinstance(res, collections.abc.Mapping):
            res = AttributeTree(res)

        return res

    def get(self,  path, *args,  is_raw_path=False,  **kwargs):
        if not is_raw_path:
            res = PathTraverser(path).apply(lambda p: self.get(p,  is_raw_path=True, **kwargs))
        else:
            item = self._mapping.get_value(path, *args, **kwargs)
            res = self._fetch(item)
        return res

    def iter(self,  path, *args, **kwargs):
        for item in self._mapping.iter(path, *args, **kwargs):
            yield self._fetch(item)


class MappingDocument(Document):

    def __init__(self, *args, root=None, mapping=None, **kwargs):
        if isinstance(root, Document):
            root = root.root
        elif not isinstance(root, Node):
            root = MappingNode(root)
        super().__init__(*args, root=root, ** kwargs)
        self._mapping = mapping

    @property
    def root(self):
        return MappingNode(super().root, mapping=self._mapping)


class MappingCollection(Collection):
    def __init__(self, uri, *args, source=None, mapping=None,  **kwargs):
        super().__init__(uri, *args, **kwargs)

        if not isinstance(mapping, Node):
            self._mapping = Document(mapping, envs=self.envs, format_type="xml").root
        else:
            self._mapping = mapping

        if not isinstance(source, Collection):
            self._source = Collection(source, envs=self.envs)
        else:
            self._source = source

    def insert_one(self, *args,  query=None, **kwargs):
        oid = self.guess_id(query or kwargs)
        doc = self._source.insert_one(_id=oid)
        return MappingDocument(fid=oid, root=doc, envs=self.envs, mapping=self._mapping)

    def find_one(self,   *args, query=None,  **kwargs):
        oid = self.guess_id(query or kwargs)
        doc = self._source.find_one(_id=oid)
        return MappingDocument(fid=oid, root=doc, envs=self.envs, mapping=self._mapping)


__SP_EXPORT__ = MappingCollection
