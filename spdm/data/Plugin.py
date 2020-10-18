import collections
import pathlib

from spdm.util.sp_export import sp_find_module
from spdm.util.urilib import urisplit
from spdm.util.logger import logger

associations = {
    "mapping": "Mapping",

    "bin": "Binary",

    "h5": "HDF5",
    "hdf5": "HDF5",

    "nc": "NetCDF",
    "netcdf": "NetCDF",

    "mds": "MDSplus",
    "mdsplus": "MDSplus",

    "namelist": "NameList",
    "nml": "NameList",

    "xml": "XML",

    "json": "JSON",
    "yaml": "YAML",
    "txt": "TXT",
    "csv": "CSV",
    "numpy": "NumPy",

    "mongo": "MongoDB",
    "mongodb": "MongoDB",
}


def find_plugin(desc, *args, pattern="{name}", fragment=None, **kwargs):

    plugin_name = None
    if isinstance(desc, collections.abc.Mapping):
        plugin_name = desc.get("schema", None)
    elif isinstance(desc, str):
        o = urisplit(desc)
        if o.schema not in [None, 'local', 'file']:
            plugin_name = o.schema
        else:
            plugin_name = file_associations.get(pathlib.Path(o.path).suffix, None)
            if plugin_name[0] == '.':
                plugin_name = plugin_name[1:]

    if plugin_name is None:
        raise ValueError(f"illegal plugin description! [{desc}]")

    pname = associations.get(plugin_name, plugin_name)

    if isinstance(pname, str):
        fname = f"{pname}{fragment}" if fragment is not None else None
        plugin = sp_find_module(pattern.format(name=pname), fragment=fname)
    elif callable(pname):
        plugin = pname(fragment)

    if plugin is None:
        raise ModuleNotFoundError(f"Can not find plugin {plugin_name}#{fragment}!  [{desc}]")
    else:
        logger.info(f"Load Plugin: {plugin.__name__}")

    return plugin
