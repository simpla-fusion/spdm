import os
import pathlib

from spdm.util.logger import logger
from spdm.util.urilib import urisplit, uriunsplit

from ..Collection import Collection
from .PluginMapping import MappingCollection


class EASTCollection(MappingCollection):
    DEVICE_NAME = "east"

    def __init__(self, uri, *args, mapping=None,   **kwargs):
        if isinstance(uri, str):
            uri = urisplit(uri)

        path = getattr(uri, "path", None) or pathlib.Path.home()/f"public_data/~t/imas/3"

        source = Collection(uriunsplit("mdsplus", uri.authority, path, None, uri.fragment),
                            *args, **kwargs)

        if mapping is None:
            mapping = []

        EAST_MAPPING_DIR = os.environ.get(
            "EAST_MAPPING_DIR",
            (pathlib.Path(__file__)/"../../../../mapping/EAST").resolve()
        )

        mapping.extend([f"{EAST_MAPPING_DIR}/imas/3/static/config.xml",
                        f"{EAST_MAPPING_DIR}/imas/3/dynamic/config.xml"])

        super().__init__(uri, source=source,  id_hasher="{shot}", mapping=mapping)


__SP_EXPORT__ = EASTCollection
