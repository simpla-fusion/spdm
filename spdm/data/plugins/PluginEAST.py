import pathlib
import os
from spdm.util.logger import logger
from spdm.util.urilib import urisplit

from .PluginMapping import MappingCollection


class EASTCollection(MappingCollection):
    DEVICE_NAME = "east"

    def __init__(self, uri, *args, mapping=None, tree_name=None, **kwargs):
        if isinstance(uri, str):
            uri = urisplit(uri)

        tree_name = tree_name or uri.fragment or EASTCollection.DEVICE_NAME

        authority = getattr(uri, "authority", "")

        path = getattr(uri, "path", None) or pathlib.Path.home()/f"public_data/{tree_name}/imas/3"

        source = f"mdsplus://{authority}{path}#{tree_name}"

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
