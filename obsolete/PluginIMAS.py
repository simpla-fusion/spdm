from .PluginXML import XMLHandler, XMLHolder
from .PluginHDF5 import HDF5Handler, connect_hdf5
from ..Handler import Handler
import collections
import pathlib

from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.PathTraverser import PathTraverser

from ..connect import connect
from ..Document import Document
from ..Collection import Collection


class IMASHandler(Handler):
    def __init__(self, target, *args, mapping=None, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(mapping, LazyProxy):
            self._xml_holder = mapping.__object__
            self._xml_handler = mapping.__handler__
        elif isinstance(mapping, XMLHolder):
            self._xml_holder = mapping
            self._xml_handler = XMLHandler()
        else:
            self._xml_holder = XMLHolder(mapping)
            self._xml_handler = XMLHandler()

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
            return LazyProxy(holder, handler=IMASHandler(self._target, mapping=item))
        elif isinstance(item, collections.abc.Mapping) and "{http://fusionyun.org/schema/}mdsplus" in item:
            return self._target.get(holder, item["{http://fusionyun.org/schema/}mdsplus"], *args, **kwargs)
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
   


class IMASCollection(Collection):
    def __init__(self, uri, *args, data_source=None, mapping=None,  **kwargs):

        super().__init__()

        eself._data_source = data_source or {}

    def add_data_source(self, d):
        pass

    def open_document(self, fid, mode):
        return Document(root=MDSplusHolder(self._tree_name, fid, mode="NORMAL"),
                        handler=self._handler,
                        collection=self)


def connect_imas(uri, *args, mapping=None, backend="HDF5", **kwargs):

    def id_hasher(collect, d, auto_inc=False):
        pattern = "{shot:08}_{run}"
        s = d.get("shot", 0)
        r = d.get("run", None)

        return s

    if mapping is not None:
        def wrapper(target, mfiles=mapping):
            return IMASHandler(target, mapping=mfiles)
    else:
        wrapper = None

    return connect(backend, *args, id_hasher=id_hasher, request_proxy=wrapper,  ** kwargs)


__SP_EXPORT__ = connect_imas
