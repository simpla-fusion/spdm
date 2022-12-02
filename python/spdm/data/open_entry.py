import collections.abc
from dataclasses import dataclass
from typing import Sequence, TypeVar, Union

from ..util.logger import logger
from ..util.uri_utils import URITuple, uri_merge, uri_split
from .Collection import Collection
from .Entry import Entry
from .File import File
from .FileCollection import FileCollection
from .Mapping import Mapping
from .Connection import Connection


def open_collection(uri, *args, **kwargs):
    uri = uri_split(uri)

    if uri.protocol is "localdb":
        return FileCollection(uri, *args, **kwargs)
    else:
        return Collection(uri, *args, **kwargs)


def open_entry(uri: Union[str, URITuple], *args, mapping_path="",   **kwargs) -> Entry:
    """ 
    Example:
      entry=open_entry("file+mdsplus[EAST]:///home/salmon/workspace/data/~t/?tree_name=efit_east,shot=38300")
    """

    uri = uri_split(uri)

    logger.debug(uri)

    if uri.protocol is None:
        uri.protocol = "local"

    uri.protocol = uri.protocol.lower()

    if uri.protocol in ("http", "https"):
        raise NotImplementedError(f"TODO: Access to remote files [{uri.protocol}] is not yet implemented! {data}")
    elif uri.protocol in ("ssh", "scp"):
        raise NotImplementedError(f"TODO: Access to remote files [{uri.protocol}] is not yet implemented! {data}")
    elif uri.protocol in ("file", "local"):
        entry = File(uri, *args, **kwargs).entry

    else:
        conn = open_collection(uri, *args, **kwargs)
        entry = conn.find(query=uri.query, fragment=uri.fragment)
        # raise NotImplementedError(f"Unsupported uri protocol!  {uri} ")

    if mapping_path is not None:
        mapping = Mapping(mapping_path)
        entry = mapping.map(entry, source_schema=uri.schema)

    return entry
