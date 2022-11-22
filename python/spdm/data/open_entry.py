from .Mapping import Mapping
from .Entry import Entry
from .File import File
from .Collection import Collection
from ..util.urilib import urisplit, URITuple
from dataclasses import dataclass
from typing import Sequence, TypeVar, Union
import collections.abc
from ..logger import logger


def open_entry(uri: Union[str, URITuple], *args, mapping_path="", mode="r",  **kwargs) -> Entry:
    """ 
    Example:
      entry=open_entry("file+mdsplus[EAST]:///home/salmon/workspace/data/~t/?tree_name=efit_east,shot=38300")
    """
    uri = urisplit(uri)

    if uri.protocol is None or uri.protocol in ["File", "file"]:
        entry = File(uri, *args, mode=mode, **kwargs).entry
    else:
        entry = None
        raise NotImplementedError(f"Unsupported uri prorocol!  {uri} ")

    if mapping_path is not None:
        mapping = Mapping(mapping_path)
        entry = mapping.map(entry, source_schema=uri.schema)
        
    return entry
