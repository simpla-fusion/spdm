import collections.abc
from dataclasses import dataclass
from typing import Sequence, TypeVar, Union

from ..util.logger import logger
from ..util.urilib import URITuple, urisplit
from .Collection import Collection
from .Connection import Connection
from .Database import Database
from .Entry import Entry
from .File import File
from .Mapping import Mapping


def open_connection(uri, *args, **kwargs):
    return Connection(urisplit(uri), *args, **kwargs)


def open_entry(uri: Union[str, URITuple], *args, mapping_path="",   **kwargs) -> Entry:
    """ 
    Example:
      entry=open_entry("file+mdsplus[EAST]:///home/salmon/workspace/data/~t/?tree_name=efit_east,shot=38300")
    """

    uri = urisplit(uri)

    logger.debug(uri)

    if uri.protocol is None or uri.protocol.lower() in ["file", "local"]:
        entry = File(uri, *args, **kwargs).entry
    elif uri.protocol in ("http", "https", "ssh"):
        raise NotImplementedError(
            f"TODO: Access to remote files [{uri.protocol}] is not yet implemented!")
    else:
        conn = open_connection(uri, *args, **kwargs)
        entry = conn.find(query=uri.query, fragment=uri.fragment)
        # raise NotImplementedError(f"Unsupported uri prorocol!  {uri} ")

    if mapping_path is not None:
        mapping = Mapping(mapping_path)
        entry = mapping.map(entry, source_schema=uri.schema)

    return entry
