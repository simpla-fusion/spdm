import collections
import contextlib
import pathlib
import shutil
import uuid

from ..util.logger import logger
from ..util.urilib import urisplit
from ..util.SpObject import SpObject
from .Document import Document


class DataBase(Document):
    """ 
        Default entry for database-like object
    """
    associtaion = {
        "db.local_file": ".data.db.FileCollection#FileCollection",
    }

    def __new__(cls,  metadata=None, *args, **kwargs):
        if cls is not DataBase:
            return super(DataBase, cls).__new__(cls, metadata, *args, **kwargs)

        if metadata is not None and not isinstance(metadata, collections.abc.Mapping):
            metadata = {"path": metadata}
        metadata = collections.ChainMap(metadata, kwargs)
        n_cls = metadata.get("$class", None)

        if not n_cls:
            file_format = metadata.get("format", None)
            if not file_format:
                path = pathlib.Path(metadata.get("path", ""))

                if not path.suffix:
                    raise ValueError(f"Can not guess file format from path! {path}")
                file_format = path.suffix[1:]

            n_cls = f"file.{file_format.lower()}"
            metadata["$class"] = DataBase.associtaion.get(n_cls, None) or n_cls

        n_cls = SpObject.find_class(metadata)
        if issubclass(n_cls, cls):
            return object.__new__(n_cls)
        else:
            return n_cls(metadata, *args, **kwargs)

    def __init__(self, uri=None, *args,  **kwargs):
        super().__init__(*args, path=uri, ** kwargs)
        self._path = urisplit(uri)
        logger.debug(f"Loading DataBase [{self.__class__.__module__}.{self.__class__.__name__}] : {uri}")
