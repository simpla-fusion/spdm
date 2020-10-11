import collections
import re
import urllib
import pathlib
from typing import Any, Dict, List, NewType, Tuple
import numpy
from spdm.util.logger import logger
from spdm.util.utilities import whoami
from .Document import Document

InsertOneResult = collections.namedtuple("InsertOneResult", "inserted_id success")
InsertManyResult = collections.namedtuple("InsertManyResult", "inserted_ids success")
UpdateResult = collections.namedtuple("UpdateResult", "upserted_id success")
DeleteResult = collections.namedtuple("DeleteResult", "deleted_id success")


class Collection(object):
    ''' Collection of documents
    '''

    def __init__(self, *args,  **kwargs):
        super().__init__()

    def create(self, *args, **kwargs):
        return self.insert_one(*args, **kwargs).entry

    def open(self, *args, **kwargs):
        return self.find_one(*args, **kwargs).entry

    def find_one(self, predicate=None, *args, **kwargs):
        raise NotImplementedError(whoami(self))

    def find(self, predicate=None, projection=None, *args, **kwargs):
        raise NotImplementedError(whoami(self))

    def insert_one(self, document, *args, **kwargs) -> InsertOneResult:
        raise NotImplementedError(whoami(self))

    def insert_many(self, documents, *args, **kwargs) -> InsertManyResult:
        return [self.insert_one(doc, *args, **kwargs) for doc in documents]

    def replace_one(self, predicate, replacement,  *args, **kwargs) -> UpdateResult:
        raise NotImplementedError(whoami(self))

    def update_one(self, predicate, update,  *args, **kwargs) -> UpdateResult:
        raise NotImplementedError(whoami(self))

    def update_many(self, predicate, updates: list,  *args, **kwargs) -> UpdateResult:
        return [self.update_one(predicate, update, *args, **kwargs) for update in updates]

    def delete_one(self, predicate,  *args, **kwargs) -> DeleteResult:
        raise NotImplementedError(whoami(self))

    def delete_many(self, predicate, *args, **kwargs) -> DeleteResult:
        raise NotImplementedError(whoami(self))

    def count(self, predicate=None,*args, **kwargs) -> int:
        raise NotImplementedError(whoami(self))

    ######################################################################
    # TODO(salmon, 2019.07.01) support index

    def create_indexes(self, indexes: List[str], session=None, **kwargs):
        raise NotImplementedError(whoami(self))

    def create_index(self, keys: List[str], session=None, **kwargs):
        raise NotImplementedError(whoami(self))

    def ensure_index(self, key_or_list, cache_for=300, **kwargs):
        raise NotImplementedError(whoami(self))

    def drop_indexes(self, session=None, **kwargs):
        raise NotImplementedError(whoami(self))

    def drop_index(self, index_or_name, session=None, **kwargs):
        raise NotImplementedError(whoami(self))

    def reindex(self, session=None, **kwargs):
        raise NotImplementedError(whoami(self))

    def list_indexes(self, session=None):
        raise NotImplementedError(whoami(self))


class FileCollection(Collection):

    def __init__(self, path, *args, filename_pattern=None, mode="rw", document_factory=None,  **kwargs):
        
        super().__init__(*args, **kwargs)

        self._mode = mode

        self._path = pathlib.Path(path).resolve().expanduser()

        self._filename_pattern = filename_pattern or "{_id}"

        self._document_factory = document_factory

        if not self._path.parent.exists():
            if "w" not in mode:
                raise RuntimeError(f"Can not make dir {self._path}")
            else:
                self._path.parent.mkdir()
        elif not self._path.parent.is_dir():
            raise NotADirectoryError(self._path.parent)

        logger.debug(f"Open Collection : {self._path}")

    def create_document(self, fname, mode):
        fpath = self._path.with_name(fname)
        logger.debug(f"Opend Document: {fpath} mode=\"{mode}\"")
        return self._document_factory(fpath, mode)

    # mode in ["", auto_inc  , glob ]
    def get_filename(self, d, mode=""):
        if callable(self._filename_pattern):
            fname = self._filename_pattern(self._path, d, mode)
        elif not isinstance(self._filename_pattern, str):
            raise TypeError(self._filename_pattern)
        elif mode == "auto_inc":
            fnum = len(list(self._path.parent.glob(self._path.name.format(_id="*"))))
            fname = (self._path.name or self._filename_pattern).format(_id=fnum)
        elif mode == "glob":
            fname = (self._path.name or self._filename_pattern).format(_id="*")
        else:
            try:
                fname = (self._path.name or self._filename_pattern).format_map(d)
            except KeyError:
                fname = None

        return fname

    def insert_one(self, data=None,  **kwargs):
        doc = self.create_document(self.get_filename(data or kwargs, mode="auto_inc"), mode="w")
        doc.update(data or kwargs)
        return doc

    def find_one(self, predicate=None, projection=None, **kwargs):
        fname = self.get_filename(predicate or kwargs)
        doc = None
        if fname is not None:
            doc = self.create_document(fname, mode="r")
        else:
            for fp in self._path.parent.glob(self.get_filename(predicate or kwargs, mode="glob")):
                if not fp.exists():
                    continue
                doc = self.create_document(fname, mode="r")
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

    def count(self, predicate=None,   *args, **kwargs) -> int:
        raise NotImplementedError()
