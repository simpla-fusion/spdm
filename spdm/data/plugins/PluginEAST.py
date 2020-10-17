import pathlib
from spdm.util.logger import logger
from ..Handler import HandlerProxy


def connect_east(uri, *args, mapping=None, **kwargs):

    path = getattr(uri, "path", uri)
    mapping_files = ["/home/salmon/workspace/SpDev/SpDB/mapping/EAST/imas/3/static",
                     "/home/salmon/workspace/SpDev/SpDB/mapping/EAST/imas/3/dynamic"]

    return connect("imas://", backend=f"hdf5://{uri.path}", * args,   ** kwargs)


__SP_EXPORT__ = connect_imas_east
