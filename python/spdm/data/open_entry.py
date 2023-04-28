import typing

from ..utils.misc import fetch_request
from ..utils.uri_utils import URITuple, uri_split
from .Collection import Collection
from .Entry import Entry
from .File import File
from .FileCollection import FileCollection
from .Mapper import create_mapper


def open_db(uri: typing.Union[str, URITuple], *args,
            source_schema=None,
            target_schema=None, mapper=None, ** kwargs) -> Collection:
    uri = uri_split(uri)
    if uri.protocol is None:
        uri.protocol = "localdb"

    if source_schema is None and uri.schema != "":
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


def open_entry(uri: typing.Union[str, URITuple], *args,
               source_schema=None, target_schema=None,
               mapper=None, ** kwargs) -> Entry:
    """
    Example:
      entry=open_entry("file+mdsplus[EAST]:///home/salmon/workspace/data/~t/?tree_name=efit_east,shot=38300")
    """

    uri = uri_split(uri)

    if source_schema is None:
        source_schema = uri.schema

    if source_schema is not None and source_schema != target_schema:
        mapper = create_mapper(mapper, source_schema=source_schema, target_schema=target_schema)
    else:
        mapper = None

    if uri.protocol in ("http", "https"):
        return Entry(fetch_request(uri))

    elif uri.protocol in ("file", "local", "ssh", "scp", None):
        entry = File(uri, *args, **kwargs).entry
        if mapper is not None:
            entry = mapper.map(entry)
        return entry
    else:
        db = open_db(uri, *args, mapper=mapper, **kwargs)
        return db.find_one(uri.query, fragment=uri.fragment)
