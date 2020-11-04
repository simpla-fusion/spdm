import collections
import pathlib

from spdm.util.sp_export import sp_find_module
from spdm.util.urilib import urisplit, URISplitResult
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
    "gfile": "GEQdsk"
}


def find_plugin(desc, *args, pattern="{name}", fragment=None, **kwargs):

    schema = ""

    if isinstance(desc, collections.abc.Mapping):
        schema = desc.get("schema", "")

    else:
        if isinstance(desc, str):
            o = urisplit(desc)
        elif isinstance(desc, URISplitResult):
            o = desc
        else:
            raise TypeError(f"Illegal uri type! {desc}")
        
        if o.schema not in [None, 'local', 'file']:
            schema = o.schema
        else:
            suffix = pathlib.Path(o.path).suffix
            if suffix[0] == '.':
                suffix = suffix[1:]
            schema = associations.get(suffix, None)

    plugin_name = schema.split('+')[0]

    if plugin_name is None:
        raise ValueError(f"illegal plugin description! [{desc}]")

    pname = associations.get(plugin_name, plugin_name)

    if isinstance(pname, str):
        def _load_mod(n):
            return sp_find_module(pattern.format(name=n), fragment=f"{n}{fragment}" if fragment is not None else None)

        plugin = _load_mod(pname) or _load_mod(pname.capitalize()) or \
            _load_mod(pname.upper()) or _load_mod(pname.lower())

    elif callable(pname):
        plugin = pname(fragment)

    if plugin is None:
        raise ModuleNotFoundError(f"Can not find plugin {plugin_name}#{fragment}!  [{desc}]")
    else:
        logger.info(f"Load Plugin: {plugin.__name__}")

    return plugin
