import pathlib
from spdm.util.logger import logger

from ..connect import connect


def connect_imas_hdf5(uri, *args,  **kwargs):

    path = getattr(uri, "path", uri)

    return connect("imas://", backend=f"hdf5://{path}", * args,   ** kwargs)


__SP_EXPORT__ = connect_imas_hdf5
