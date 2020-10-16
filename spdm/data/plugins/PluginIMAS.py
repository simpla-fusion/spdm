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
            return Request(path).apply(lambda p: self.get(holder, p, is_raw_path=True, **kwargs))
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

    def get(self, holder, path, *args,  is_raw_path=False,  **kwargs):
        if not is_raw_path:
            return Request(path).apply(lambda p: self.get(holder, p,  is_raw_path=True, **kwargs))
        else:
            req = self._xml_handler.get_value(self._xml_holder, path, *args, **kwargs)

            if req is None:
                res = self._target.get(holder, path,  *args,  **kwargs)
            elif isinstance(req, collections.abc.Mapping):
                m_req = req.get("{http://hpc.ipp.ac.cn/SpDB}mdsplus", None)
                if m_req is None:
                    res = req
                else:
                    res = self._target.get(holder, m_req, *args, **kwargs)
            else:
                res = req

            return res

    def iter(self, holder, path, *args, **kwargs):
        for item in self._xml_handler.iter(self._xml_holder, path):
            if isinstance(item, LazyProxy) or isinstance(item, XMLHolder):
                yield LazyProxy(holder, handler=IMASHandler(self._target, mapping=item))
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


def connect_imas(uri, *args, mapping=None, backend="HDF5", **kwargs):

    def id_pattern(collect, d, auto_inc=False):
        pattern = "{shot:08}_{run}"
        s = d.get("shot", 0)
        r = d.get("run", None)

        return s

    if mapping is not None:
        def wrapper(target, mfiles=mapping):
            return IMASHandler(target, mapping=mfiles)
    else:
        wrapper = None

    return connect(backend, *args, id_pattern=id_pattern, handler_proxy=wrapper,  ** kwargs)


__SP_EXPORT__ = connect_imas
