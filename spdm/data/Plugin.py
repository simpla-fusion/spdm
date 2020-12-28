import collections
import pathlib

from spdm.util.sp_export import sp_find_module
from spdm.util.urilib import urisplit
from spdm.util.logger import logger
from spdm.util.AttributeTree import AttributeTree

associations = {
    "mapping": "Mapping",
    "local": "LocalFile",
    "mongo": "MongoDB",
    "mongodb": "MongoDB",

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

    
    "gfile": "GEQdsk"
}


def find_plugin(desc, *args, pattern="{name}", fragment=None, **kwargs):

    if isinstance(desc, str):
        desc = urisplit(desc)
        logger.debug(desc)
    elif not isinstance(desc, AttributeTree):
        desc = AttributeTree(desc)

    plugin_name = ""

    if desc.schema not in [None, 'local', 'file']:
        plugin_name = desc.schema
    else:
        suffix = pathlib.Path(desc.path).suffix
        if not suffix:
            plugin_name = desc.path
        else:
            if suffix[0] == '.':
                suffix = suffix[1:]
            plugin_name = associations.get(suffix, None)

    if len(plugin_name) > 0:
        plugin_name = plugin_name.split('+')[0]

    if not plugin_name:
        raise ValueError(f"illegal plugin description! [{desc}]")

    pname = associations.get(plugin_name, plugin_name)

    if isinstance(pname, str):
        def _load_mod(n):
            try:
                mod = sp_find_module(pattern.format(name=n),
                                     fragment=f"{n}{fragment}" if fragment is not None else n)
            except ModuleNotFoundError:
                mod = None
            finally:
                return mod

        plugin = _load_mod(pname) or _load_mod(pname.capitalize()) or \
            _load_mod(pname.upper()) or _load_mod(pname.lower())

    elif callable(pname):
        plugin = pname(fragment)

    if plugin is None:
        raise ModuleNotFoundError(f"Can not find plugin {plugin_name}#{fragment}!  [{desc}]")
    else:
        logger.info(f"Load Plugin: {plugin.__name__}")

    return plugin
