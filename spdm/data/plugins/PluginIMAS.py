import pathlib
from spdm.util.logger import logger
from .PluginHDF5 import (connect_hdf5, HDF5Handler)
from .PluginXML import (XMLHolder, XMLHandler)
from ..Document import Document
from ..Handler import HandlerProxy, Request
from ..connect import connect
from spdm.util.logger import logger


class IMASHandler(Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mapping = XMLHolder(mapping_files)
        self._xml_handler = IMASHandler()

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

        return super().request(xpath, query, fragment)


def connect_imas(uri, *args, mapping_files=None, backend="HDF5", **kwargs):

    def id_pattern(collect, d, auto_inc=False):
        pattern = "{shot:08}_{run}"
        s = d.get("shot", 0)
        r = d.get("run", None)

        return s

    if mapping_files is not None:
        wrapper_doc = Document(root=XMLHolder(mapping_files), handler=IMASHandler())

        def wrapper(target, proxy=wrapper_doc):
            return HandlerProxy(target, proxy=wrapper)
    else:
        wrapper = None

    return connect(backend, *args, id_pattern=id_pattern, handler_proxy=wrapper,  ** kwargs)


__SP_EXPORT__ = connect_imas
