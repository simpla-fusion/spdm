import pathlib
from spdm.util.logger import logger
from .PluginHDF5 import (connect_hdf5, HDF5Handler)
from .PluginXML import (open_xml)

from ..Handler import HandlerProxy
from ..connect import connect


def connect_imas(uri, *args, mapping_files=None, backend="HDF5", **kwargs):

    def id_pattern(collect, d, auto_inc=False):
        pattern = "{shot:08}_{run}"
        s = d.get("shot", 0)
        r = d.get("run", None)

        if r is None and auto_inc:
            r = collect.count(shot=s)

        return pattern.format(shot=s, run=str(r))

    if mapping_files is not None:
        proxy = HandlerProxy(mapper=open_xml(mapping_files))
    else:
        proxy = None

    return connect(backend, *args, id_pattern=id_pattern,  handler=proxy, ** kwargs)


__SP_EXPORT__ = connect_imas
