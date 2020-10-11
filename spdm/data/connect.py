from spdm.util.logger import logger
from spdm.util.urilib import urisplit
from spdm.util.sp_export import sp_find_module


def connect(uri, *args, **kwargs):
    uri = urisplit(uri)
    backend = sp_find_module(f"{__package__}.plugins.Plugin{uri.schema.upper()}")
    return backend(uri, *args, **kwargs)
