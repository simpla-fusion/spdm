import collections
import pathlib

from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.PathTraverser import PathTraverser

from ..Collection import Collection
from ..Document import Document
from ..Node import Node, Handler, Holder


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
        if isinstance(mapping, collections.abc.Sequence):
            logger.debug(mapping)
            self._mapping = Document(mapping, format_type="xml")
        elif isinstance(mapping, Node):
            self._mapping = mapping
        else:
            raise TypeError(mapping)
        self._target = target

    def put(self, holder, path, value, *args, is_raw_path=False,   **kwargs):
        if not is_raw_path:
            return PathTraverser(path).apply(lambda p: self.get(holder, p, is_raw_path=True, **kwargs))
        else:
            req = self._mapping.handler.get(self._mapping.holder, path, *args, **kwargs)
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
        elif isinstance(item, LazyProxy):
            return LazyProxy(holder, handler=MappingHandler(self._target, mapping=Node(item.__object__, handler=item.__handler__)))
        elif isinstance(item, Holder):
            return LazyProxy(holder, handler=MappingHandler(self._target, mapping=Node(item, handler=self._mapping.handler)))
        elif isinstance(item, collections.abc.Mapping) and "{http://hpc.ipp.ac.cn/SpDB}mdsplus" in item:
            if self._target is None:
                raise RuntimeError()
            return self._target.get(holder, item["{http://hpc.ipp.ac.cn/SpDB}mdsplus"], *args, **kwargs)
        else:
            return item

    def get(self, holder, path, *args,  is_raw_path=False,  **kwargs):
        if not is_raw_path:
            res = PathTraverser(path).apply(lambda p: self.get(holder, p,  is_raw_path=True, **kwargs))
        else:
            item = self._mapping.get_value(path, *args, **kwargs)
            res = self._fetch_from_xml(holder, item)
            if res is None:
                res = self._target.get(holder, path,  *args,  **kwargs)
        return res

    def iter(self, holder, path, *args, **kwargs):
        for item in self._mapping.iter(path):
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
        return Document(holder=src.holder,  handler=MappingHandler(src.handler, mapping=self._mapping))

    def insert_one(self, pred=None, *args, **kwargs):
        doc = self._source.insert_one(_id=self.guess_id(pred or kwargs))
        return self._do_wrap(doc)

    def find_one(self, pred=None, *args, **kwargs):
        return self._do_wrap(self._source.find_one(_id=self.guess_id(pred or kwargs)))


__SP_EXPORT__ = MappingCollection
