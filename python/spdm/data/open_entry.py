import collections.abc
from dataclasses import dataclass
from typing import Sequence, TypeVar, Union

from ..util.logger import logger
from ..util.uri_utils import URITuple, uri_merge, uri_split
from .Entry import Entry
from .File import File
from .Mapper import create_mapper, Mapper
from ..common.tags import _not_found_, _undefined_
from .Collection import Collection
from .FileCollection import FileCollection


def open_db(uri: Union[str, URITuple], *args,
            source_schema=_undefined_,
            target_schema=_undefined_, mapper=None, ** kwargs) -> Collection:
    uri = uri_split(uri)
    if uri.protocol is None:
        uri.protocol = "localdb"

    if source_schema is _undefined_ and uri.schema != "":
        source_schema = uri.schema

    if source_schema == target_schema:
        mapper = None
    else:
        mapper = create_mapper(mapper, source_schema=source_schema, target_schema=target_schema)

    if uri.protocol == "localdb":
        db = FileCollection(uri, *args, mapper=mapper, **kwargs)
    else:
        db = Collection(uri, *args, mapper=mapper, **kwargs)
    return db


def open_entry(uri: Union[str, URITuple], *args,
               source_schema=_undefined_, target_schema=_undefined_,
               mapper=None, ** kwargs) -> Union[Entry, Collection]:
    """
    Example:
      entry=open_entry("file+mdsplus[EAST]:///home/salmon/workspace/data/~t/?tree_name=efit_east,shot=38300")
    """

    uri = uri_split(uri)

    if source_schema is _undefined_:
        source_schema = uri.schema

    if source_schema is not None and source_schema != target_schema:
        mapper = create_mapper(mapper, source_schema=source_schema, target_schema=target_schema)
    else:
        mapper = None

    if uri.protocol in ("http", "https"):
        raise NotImplementedError(f"TODO: Access to remote files [{uri.protocol}] is not yet implemented!")
    # elif uri.protocol in ():
    #     raise NotImplementedError(f"TODO: Access to remote files [{uri.protocol}] is not yet implemented!")
    elif uri.protocol in ("file", "local", "ssh", "scp", None):
        entry = File(uri, *args, **kwargs).entry
        if mapper is not None:
            entry = mapper.map(entry)
        return entry
    else:
        db = open_db(uri, *args, mapper=mapper, **kwargs)
        return db.find_one(uri.query, fragment=uri.fragment)
