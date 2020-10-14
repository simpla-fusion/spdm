import pathlib
from spdm.util.logger import logger
from .PluginHDF5 import (connect_hdf5, HDF5Handler)
from .PluginXML import (open_xml)

from ..Handler import HandlerProxy
from ..connect import connect
from spdm.util.logger import logger


def connect_imas(uri, *args, mapping_files=None, backend="HDF5", **kwargs):

    def id_pattern(collect, d, auto_inc=False):
        pattern = "{shot:08}_{run}"
        s = d.get("shot", 0)
        r = d.get("run", None)

        return s

    def uri_mapper(path):
        xpath = ""
        query = {}
        prev = None
        for p in path:
            if type(p) is int and prev == "time_slice":
                query["itime"] = p
            elif type(p) is int:
                xpath += f"[{p+1}]"
            elif isinstance(p, str):
                xpath += f"/{p}"
            else:
                # TODO: handle slice
                raise TypeError(f"Illegal path type! {type(p)} {path}")
            prev = p

        if len(xpath) > 0 and xpath[0] == "/":
            xpath = xpath[1:]

        return xpath, query

    if mapping_files is not None:
        wrapper_doc = open_xml(mapping_files, mapper=uri_mapper)

        def wrapper(handler, wrapper=wrapper_doc):
            return HandlerProxy(handler, wrapper=wrapper)
    else:
        wrapper = None

    return connect(backend, *args, id_pattern=id_pattern, handler_wrapper=wrapper,  ** kwargs)


__SP_EXPORT__ = connect_imas
