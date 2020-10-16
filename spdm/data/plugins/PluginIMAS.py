import pathlib
import collections
from spdm.util.logger import logger
from .PluginHDF5 import (connect_hdf5, HDF5Handler)
from .PluginXML import (XMLHolder, XMLHandler)
from ..Document import Document
from ..Handler import (Request, Handler)
from ..connect import connect
from spdm.util.logger import logger
from spdm.util.LazyProxy import LazyProxy


class IMASHandler(Handler):
    def __init__(self, target, *args, mapping_files=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._xml_holder = XMLHolder(mapping_files)

        # class _Handler(XMLHandler):
        #     def request(self, path):
        #         xpath = []
        #         prev = None
        #         for p in path:
        #             if prev == "time_slice":
        #                 xpath.append("@id='*'")
        #             else:
        #                 xpath.append(p)
        #             prev = p
        #         return super().request(xpath)

        self._xml_handler = XMLHandler()

        self._target = target

    def put(self, holder, path, value, *args, is_raw_path=False,   **kwargs):
        if not is_raw_path:
            return Request(path).apply(lambda p: self.get(holder, p, is_raw_path=True, **kwargs))
        else:
            req = self._xml_handler.get(self._xml_holder, path, *args, **kwargs)
            if isinstance(req, str):
                self._target.put(holder, req, value, is_raw_path=True)
            elif isinstance(res, collections.abc.Sequence):
                raise NotImplementedError()
            elif isinstance(res, collections.abc.Mapping):
                raise NotImplementedError()

                # xpath, query, fragment=req
                # if isinstance(query, collections.abc.Mapping):
                #     return self._target.put(holder,  xpath.format_map(query),  value)
                # elif isinstance(query, collections.abc.Iterable):
                #     return [self._target.put(holder, xpath.format_map(q), value) for q in query]
                # else:
                #     return self._target.put(holder,  xpath, value)
            elif req is not None:
                raise RuntimeError(f"Can not write to non-empty entry! {path}")

    def get(self, holder, path, *args,  is_raw_path=False,  **kwargs):
        if not is_raw_path:
            return Request(path).apply(lambda p: self.get(holder, p,  is_raw_path=True, **kwargs))
        else:
            res = self._xml_handler.get_value(self._xml_holder, path, *args, **kwargs)

            if isinstance(res, str):
                res = self._target.get(holder, res, *args, is_raw_path=True,  **kwargs)
            elif isinstance(res, collections.abc.Sequence):
                raise NotImplementedError()
            if isinstance(res, collections.abc.Mapping):
                res = self._target.get(holder, res, *args,   **kwargs)
            elif res is None:
                res = self._target.get(holder, path,  *args,  **kwargs)

            return res

    def iter(self, holder, path, *args, **kwargs):
        req = self._xml_handler.get(self._xml_holder,  path, *args, **kwargs)

        for item in self._xml_handler.iter(self._xml_holder, path):
            if isinstance(item, XMLHolder):
                yield LazyProxy(item, handler=self._xml_handler)
            else:
                yield item
        # else:
        #     spath = self._xml_handler.request(path)

        #     def r_iter(spath, query):
        #         if len(query) == 0:
        #             return

        #         k, v = query[0]
        #         if not isinstance(v, slice):
        #             yield from spath.format_map({k: v}, query[1:])
        #         else:
        #             logger.debug(v)

        #             for i in range(v.start, v.stop, v.step or 1):
        #                 yield spath.format_map({k: i}, query[1:])

        #     for p in r_iter(spath, list(query.items())):
        #         logger.debug(p)

            # if count == 0:
            #     yield from self._target.iter(holder, path, *args, **kwargs)


def connect_imas(uri, *args, mapping_files=None, backend="HDF5", **kwargs):

    def id_pattern(collect, d, auto_inc=False):
        pattern = "{shot:08}_{run}"
        s = d.get("shot", 0)
        r = d.get("run", None)

        return s

    if mapping_files is not None:
        def wrapper(target, mfiles=mapping_files):
            return IMASHandler(target, mapping_files=mfiles)
    else:
        wrapper = None

    return connect(backend, *args, id_pattern=id_pattern, handler_proxy=wrapper,  ** kwargs)


__SP_EXPORT__ = connect_imas
