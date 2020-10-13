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

    def __init__(self, *args, mode="rw",  id_pattern=None,  **kwargs):
        super().__init__()
        self._mode = mode
        self._id_pattern = id_pattern

    @property
    def mode(self):
        return self._mode

    # mode in ["", auto_inc  , glob ]
    def guess_id(self, d, auto_inc=True):
        fid = None
        if callable(self._id_pattern):
            fid = self._id_pattern(self, d, auto_inc)
        elif isinstance(self._id_pattern, str):
            fid = self._id_pattern.format_map(d)

        return fid

    def create_document(self, fid, mode):
        logger.debug(f"Opend Document: {fpath} mode=\"{mode}\"")
        raise NotImplementedError(whoami(self))

    def insert(self, *args, **kwargs):
        return self.insert_one(*args, **kwargs)

    def open(self, *args, **kwargs):
        return self.find_one(*args, **kwargs)

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

    def count(self, predicate=None, *args, **kwargs) -> int:
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

    def __init__(self, path, *args,
                 file_extension=".dat",
                 file_factory=None,
                 **kwargs):

        super().__init__(*args,  **kwargs)

        if isinstance(path, str) and path.endswith("/"):
            path = f"{path}/{{_id}}{file_extension}"

        self._path = pathlib.Path(path).resolve().expanduser()

        if self._path.suffix == '':
            self._path = self._path.with_suffix(file_extension)

        if "{_id}" not in self._path.stem:
            self._path = self._path.with_name(f"{self._path.stem}{{_id}}{self._path.suffix}")

        logger.debug(self._path)

        self._file_factory = file_factory

        if not self._path.parent.exists():
            if "w" not in mode:
                raise RuntimeError(f"Can not make dir {self._path}")
            else:
                self._path.parent.mkdir()
        elif not self._path.parent.is_dir():
            raise NotADirectoryError(self._path.parent)

        logger.debug(f"Open Collection : {self._path}")

    def guess_id(self, d, auto_inc=True):
        fid = super().guess_id(d, auto_inc=auto_inc)

        if fid is None and auto_inc:
            fid = self.count()

        return fid

    def create_document(self, fid, mode=None):
        fname = self._path.name.format(_id=fid)
        fpath = self._path.with_name(self._path.name.format(_id=fid))
        logger.debug(f"Opend Document: {fpath} mode=\"{ mode or self.mode}\"")
        return self._file_factory(fpath, mode or self.mode)

    def insert_one(self, data=None, *args,  **kwargs):
        doc = self.create_document(self.guess_id(data or kwargs, auto_inc=True), mode="w")
        doc.update(data or kwargs)
        return doc

    def find_one(self, predicate=None, projection=None, **kwargs):
        fpath = self._path.with_name(self._path.name.format(_id=self.guess_id(predicate or kwargs)))

        doc = None
        if fpath.exists():
            doc = self.create_document(fpath)
        else:
            for fp in self._path.parent.glob(self._path.name.format(_id="*")):
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

        if predicate is None:
            logger.warning("NOT IMPLEMENTED! count by predicate")

        return len(list(self._path.parent.glob(self._path.name.format(_id="*"))))
