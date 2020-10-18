import collections
import pathlib

from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.PathTraverser import PathTraverser

from ..Collection import Collection
from ..Document import Document
from ..Node import Node, Handler


class MappingHandler(Handler):
    def __init__(self, target, *args, mapping=None, **kwargs):
        super().__init__(*args, **kwargs)
        # if isinstance(mapping, LazyProxy):
        #     self._xml_holder = mapping.__object__
        #     self._xml_handler = mapping.__handler__
        # elif isinstance(mapping, XMLHolder):
        #     self._xml_holder = mapping
        #     self._xml_handler = XMLHandler()
        # else:
        #     self._xml_holder = XMLHolder(mapping)
        #     self._xml_handler = XMLHandler()
        if not isinstance(mapping, Node):
            self._mapping = Document(mapping)
        else:
            self._mapping = mapping
        self._target = target

    def put(self, holder, path, value, *args, is_raw_path=False,   **kwargs):
        if not is_raw_path:
            return PathTraverser(path).apply(lambda p: self.get(holder, p, is_raw_path=True, **kwargs))
        else:
            req = self._xml_handler.get(self._xml_holder, path, *args, **kwargs)
            if isinstance(req, str):
                self._target.put(holder, req, value, is_raw_path=True)
            elif isinstance(res, collections.abc.Sequence):
                raise NotImplementedError()
            elif isinstance(res, collections.abc.Mapping):
                raise NotImplementedError()
            elif req is not None:
                raise RuntimeError(f"Can not write to non-empty entry! {path}")

    def _fetch_from_xml(self, holder, item, *args, **kwargs):
        if item is None:
            return None
        elif isinstance(item, LazyProxy) or isinstance(item, XMLHolder):
            return LazyProxy(holder, handler=MappingHandler(self._target, mapping=item))
        elif isinstance(item, collections.abc.Mapping) and "{http://hpc.ipp.ac.cn/SpDB}mdsplus" in item:
            return self._target.get(holder, item["{http://hpc.ipp.ac.cn/SpDB}mdsplus"], *args, **kwargs)
        else:
            return item

    def get(self, holder, path, *args,  is_raw_path=False,  **kwargs):
        if not is_raw_path:
            res = PathTraverser(path).apply(lambda p: self.get(holder, p,  is_raw_path=True, **kwargs))
        else:
            item = self._xml_handler.get_value(self._xml_holder, path, *args, **kwargs)
            res = self._fetch_from_xml(holder, item)
            if res is None:
                res = self._target.get(holder, path,  *args,  **kwargs)
        return res

    def iter(self, holder, path, *args, **kwargs):
        for item in self._xml_handler.iter(self._xml_holder, path):
            yield self._fetch_from_xml(holder, item, *args, **kwargs)


class MappingCollection(Collection):
    def __init__(self, uri, *args, source=None, mapping=None,  **kwargs):

        super().__init__(uri, *args, **kwargs)

        if not isinstance(mapping, Node):
            self._mapping = Document(mapping, format_type="xml")
        else:
            self._mapping = mapping

        if not isinstance(source, Collection):
            self._source = Collection(source)
        else:
            self._source = source

    def _do_wrap(self, src):
        return Document(src.holder,  handler=MappingHandler(src.handler, mapping=self._mapping))

    def insert_one(self, pred=None, *args, **kwargs):
        return self._do_wrap(self._source.insert_one(_id=self.guess_id(pred or kwargs)))

    def find_one(self, pred=None, *args, **kwargs):
        return self._do_wrap(self._source.find_one(_id=self.guess_id(pred or kwargs)))


__SP_EXPORT__ = MappingCollection
