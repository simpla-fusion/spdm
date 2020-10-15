import pathlib
from spdm.util.logger import logger
from .PluginHDF5 import (connect_hdf5, HDF5Handler)
from .PluginXML import (XMLHolder, XMLHandler)
from ..Document import Document
from ..Handler import (Request, Handler)
from ..connect import connect
from spdm.util.logger import logger


class IMASHandler(Handler):
    def __init__(self, target, *args, mapping_files=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._xml_holder = XMLHolder(mapping_files)
        self._xml_handler = XMLHandler()
        self._target = target

    def request(self, path, query={}, fragment=None):
        xpath = []
        query = {}
        prev = None
        for p in path:
            if prev == "time_slice":
                query["itime"] = p
            else:
                xpath.append(p)
            prev = p

        if len(xpath) > 0 and xpath[0] == "/":
            xpath = xpath[1:]
        return Request(xpath, query, fragment)

    def put(self, holder, path, value, *args,  **kwargs):
        obj = self._xml_handler.get(self._xml_holder, **self.request(path, *args, **kwargs)._asdict())

        if isinstance(obj, Request):
            path, query, fragment = obj
            return self._target.put(holder, path, value,  query=query, fragment=fragment)
        elif obj is not None:
            raise RuntimeError(f"Can not write to non-empty entry! {path}")
        else:
            return self._target.put(holder, path, value, *args, **kwargs)

    def get(self, holder, path, projection=None, *args, **kwargs):
        obj = self._xml_handler.get(self._xml_holder, **self.request(path, *args, **kwargs)._asdict())

        if isinstance(obj, Request):
            path, query, fragment = obj
            return self._target.get(holder,  path,  projection=projection, query=query, fragment=fragment)
        elif obj is None:
            return self._target.get(holder,  path,  projection=projection,  *args, **kwargs)
        elif projection is None:
            return obj
        else:
            raise NotImplementedError()

    def iter(self, holder, path, *args, **kwargs):

        try:
            for item in self._xml_handler.iter(self._xml_holder, **self.request(path, *args, **kwargs)._asdict()):
                if isinstance(item, Request):
                    spath, query, fragment = item
                    yield self._target.get(spath, query=query, fragment=fragment)
                else:
                    yield item
        except StopIteration:
            pass
            # yield from self._target.iter(path, *args, **kwargs)


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
