import collections
import inspect
import pathlib
import re
import urllib
from functools import cached_property
from typing import Any, Dict, List, NewType, Tuple

import numpy
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.sp_export import sp_find_module
from spdm.util.SpObject import SpObject
from spdm.util.urilib import urisplit, uriunsplit

from .Document import Document
from .File import File

InsertOneResult = collections.namedtuple("InsertOneResult", "inserted_id success")
InsertManyResult = collections.namedtuple("InsertManyResult", "inserted_ids success")
UpdateResult = collections.namedtuple("UpdateResult", "upserted_id success")
DeleteResult = collections.namedtuple("DeleteResult", "deleted_id success")


class Collection(SpObject):
    ''' Collection of documents
    '''
    DOCUMENT_CLASS = Document
    ID_TAG = "{:06}"

    associations = {
        "mapping": f"{__package__}.db.Mapping#MappingCollection",
        "local": f"{__package__}.Collection#CollectionLocalFile",

        "mds": f"{__package__}.db.MDSplus#MDSplusCollection",
        "mdsplus": f"{__package__}.db.MDSplus#MDSplusCollection",

        "mongo": f"{__package__}.db.MongoDB",
        "mongodb": f"{__package__}.db.MongoDB",

        "imas": f"{__package__}.db.IMAS#IMASCollection",

    }
    metadata = AttributeTree()

    @staticmethod
    def __new__(cls, _metadata=None, *args,   **kwargs):
        if cls is not Collection and _metadata is None:
            return object.__new__(cls)
            # return super(Collection, cls).__new__(desc, *args, **kwargs)

        if isinstance(_metadata, str):
            _metadata = {"$class": urisplit(_metadata).schema,  "path": _metadata}
        elif desc is None:
            _metadata = {}
        elif isinstance(_metadata, AttributeTree):
            _metadata = _metadata.__as_native__()

        schemas = (_metadata["$class"] or "local").split('+')
        if not schemas:
            raise ValueError(_metadata)
        elif len(schemas) > 1:
            schema = "mapping"
        else:
            schema = schemas[0]

        n_cls = Collection.associations.get(schema.lower(), f"{__package__}.db.{schema}")

        if inspect.isclass(n_cls):
            res = object.__new__(n_cls)
        else:
            res = SpObject.__new__(Collection, _metadata=n_cls)

        # else:
        #     raise RuntimeError(f"Illegal schema! {schema} {n_cls} {desc}")

        return res

    def __init__(self, uri, *args, id_hasher=None, envs=None, mode="rw", auto_inc_idx=True, doc_factory=None, **kwargs):
        super().__init__()

        logger.info(f"Open {self.__class__.__name__} : {uri}")

        self._uri = uri

        self._id_hasher = id_hasher

        self._envs = collections.ChainMap(kwargs, envs or {})

        self._mode = mode

        self._auto_inc_idx = auto_inc_idx

        self._document_factory = doc_factory or Document

    # def __del__(self):
    #     logger.info(f"Close {self.__class__.__name__}:{self._uri}")

    @cached_property
    def envs(self):
        return AttributeTree(self._envs)

    @property
    def uri(self):
        return self._uri

    @property
    def mode(self):
        return self._mode

    @property
    def is_writable(self):
        return "w" in self.mode or 'x' in self.mode

    @property
    def handler(self):
        return self._handler

    # mode in ["", auto_inc  , glob ]
    def guess_id(self, *args, **kwargs):
        fid = None
        if not self._id_hasher:
            fid = args[0] if len(args) > 0 else None
        elif callable(self._id_hasher):
            fid = self._id_hasher(self, *args, **kwargs)
        elif isinstance(self._id_hasher, str):
            try:
                fid = self._id_hasher.format(*args, **kwargs)
            except Exception:
                fid = None
        else:
            raise RuntimeError(f"Can not guess id from {args,kwargs}")
        return fid

    @property
    def next_id(self):
        raise NotImplementedError()

    def open_document(self, fid, *args, mode=None, **kwargs):
        logger.debug(f"Open Document: {fid} [mode=\"{ mode or self.mode}\"]")
        return self._document_factory(self.guess_path(fid), *args, mode=mode or self.mode, **kwargs)

    def open(self, *args, mode=None, **kwargs):
        mode = mode or self.mode
        if "w" in mode:
            return self.create(*args, **kwargs)
        elif "w" not in mode:
            return self.find_one(*args, **kwargs)
        else:
            raise RuntimeWarning("Collection is not writable!")

    def create(self, *args, **kwargs):
        return self.open_document(self.guess_id(*args, **kwargs) or self.next_id, mode="x")

    def insert(self, data, *args, **kwargs):
        if isinstance(data, list):
            return self.insert_many(data, *args, **kwargs)
        else:
            return self.insert_one(*args, **kwargs)

    def insert_one(self, data, * args,  **kwargs) -> InsertOneResult:
        doc = self.open_document(self.guess_id(*args, **kwargs) or self.next_id, mode="x")
        if data is not None:
            doc.update(data)
        return doc

    def insert_many(self, documents, *args, **kwargs) -> InsertManyResult:
        return [self.insert_one(doc, *args, **kwargs) for doc in documents]

    def find_one(self, predicate=None, *args, **kwargs):
        raise NotImplementedError()

    def find(self, predicate=None, projection=None, *args, **kwargs):
        raise NotImplementedError()

    def replace_one(self, predicate, replacement,  *args, **kwargs) -> UpdateResult:
        raise NotImplementedError()

    def update_one(self, predicate, update,  *args, **kwargs) -> UpdateResult:
        raise NotImplementedError()

    def update_many(self, predicate, updates: list,  *args, **kwargs) -> UpdateResult:
        return [self.update_one(predicate, update, *args, **kwargs) for update in updates]

    def delete_one(self, predicate,  *args, **kwargs) -> DeleteResult:
        raise NotImplementedError()

    def delete_many(self, predicate, *args, **kwargs) -> DeleteResult:
        raise NotImplementedError()

    def count(self, predicate=None, *args, **kwargs) -> int:
        raise NotImplementedError()

    ######################################################################
    # TODO(salmon, 2019.07.01) support index

    def create_indexes(self, indexes: List[str], session=None, **kwargs):
        raise NotImplementedError()

    def create_index(self, keys: List[str], session=None, **kwargs):
        raise NotImplementedError()

    def ensure_index(self, key_or_list, cache_for=300, **kwargs):
        raise NotImplementedError()

    def drop_indexes(self, session=None, **kwargs):
        raise NotImplementedError()

    def drop_index(self, index_or_name, session=None, **kwargs):
        raise NotImplementedError()

    def reindex(self, session=None, **kwargs):
        raise NotImplementedError()

    def list_indexes(self, session=None):
        raise NotImplementedError()


class CollectionLocalFile(Collection):

    def __init__(self, uri, *args,
                 file_extension=".dat",
                 doc_factory=None,
                 **kwargs):

        super().__init__(uri, *args, doc_factory=doc_factory or File, **kwargs)

        if isinstance(uri, str):
            uri = urisplit(uri)

        path = getattr(uri, "path", "./")  # .replace("*", Collection.ID_TAG)

        if "*" not in path:  # not path.endswith(file_extension):
            path = f"{path}*{file_extension}"

        pos = path.rfind('/', 0, path.find('*'))

        self._path = pathlib.Path(path[:pos]).resolve().expanduser()

        self._doc_name = path[pos+1:]

        # if self._path.suffix == '':
        #     self._path = self._path.with_suffix(file_extension)

        # if Collection.ID_TAG not in self._path.stem:
        #     self._path = self._path.with_name(f"{self._path.stem}{Collection.ID_TAG}{self._path.suffix}")
        if not self._path.exists():
            if "w" not in self._mode:
                raise RuntimeError(f"Can not make dir {self._path}")
            else:
                self._path.mkdir()
        elif not self._path.parent.is_dir():
            raise NotADirectoryError(self._path)

        logger.debug((self._path, self._doc_name))

    @property
    def next_id(self):
        return len(list(self._path.glob(self._doc_name)))

    def guess_path(self, f_id):
        return self._path/(self._doc_name.replace('*', Collection.ID_TAG)).format(f_id)

    def find_one(self, predicate=None, projection=None, **kwargs):
        f_path = self.guess_filepath(**collections.ChainMap(predicate or {}, kwargs))

        doc = None
        if f_path.exists():
            doc = self.open_document(f_path)
        else:
            for fp in self._path.parent.glob(self._path.name.format(_id="*")):
                if not fp.exists():
                    continue
                doc = self.open_document(fp, mode="r")
                if doc.check(predicate):
                    break
                else:
                    doc = None

        if projection is not None:
            raise NotImplementedError()

        return doc

    def update_one(self, predicate, update,  *args, **kwargs):
        raise NotImplementedError()

    def delete_one(self, predicate,  *args, **kwargs):
        raise NotImplementedError()
