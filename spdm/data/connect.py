from spdm.util.logger import logger
from spdm.util.urilib import urisplit
from spdm.util.sp_export import sp_find_module


def connect(uri, *args, **kwargs):
    if isinstance(uri, str):
        uri = urisplit(uri)

    schema = getattr(uri, "schema", None)

    if schema is None:
        raise ValueError(f"Illegal uri! {uri}")

    plugin_name = schema.replace('+', '_').upper()

    backend = sp_find_module(f"{__package__}.plugins.Plugin{plugin_name}")

    if backend is None:
        raise ModuleNotFoundError(uri)
    
    logger.info(f"Load SpDB.Plugin: {plugin_name}")
    
    return backend(uri, *args, **kwargs)
