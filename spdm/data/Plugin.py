import collections
import pathlib

from spdm.util.sp_export import sp_find_module
from spdm.util.urilib import urisplit
from spdm.util.logger import logger

file_associations = {
    ".bin": "binary",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".nc": "netcdf",
    ".netcdf": "netcdf",
    ".namelist": "namelist",
    ".nml": "namelist",
    ".namelist": "namelist",
    ".json": "_json",
    ".yaml": "_yaml",
    ".txt": "txt",
    ".csv": "csv",
    ".numpy": "numpy",
    ".xml": "xml"
}

plugin_spec = collections.namedtuple("plugin_spec", "Description Collection Document ")


def find_plugin(desc, *args, pattern="{name}", fragment=None, **kwargs):

    schema = None
    if isinstance(desc, collections.abc.Mapping):
        schema = schema.get("schema", None)
    elif isinstance(desc, str):
        o = urisplit(desc)
        if o.schema not in [None, 'local', 'file']:
            schema = o.schema
        else:
            schema = file_associations.get(pathlib.Path(o.path).suffix, None)

    if schema is None:
        raise ValueError(f"illegal plugin description! [{desc}]")

    plugin_name = pattern.format(name=schema.upper())
    if fragment is not None:
        fragment = fragment.format(name=schema.capitalize())

    plugin = sp_find_module(plugin_name, fragment=fragment)

    if plugin is None:
        raise ModuleNotFoundError(f"Can not find plugin {plugin_name}#{fragment}!  [{desc}")
    else:
        logger.info(f"Load Plugin: {plugin.__name__}")

    return plugin
