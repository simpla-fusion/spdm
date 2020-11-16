import collections
import inspect
import pathlib
import re
import urllib
from typing import Any, Dict, List, NewType, Tuple

import numpy
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.sp_export import sp_find_module
from spdm.util.urilib import urisplit

from .Document import Document
from .Plugin import find_plugin

InsertOneResult = collections.namedtuple("InsertOneResult", "inserted_id success")
InsertManyResult = collections.namedtuple("InsertManyResult", "inserted_ids success")
UpdateResult = collections.namedtuple("UpdateResult", "upserted_id success")
DeleteResult = collections.namedtuple("DeleteResult", "deleted_id success")


class Collection(object):
    ''' Collection of documents
    '''
    DOCUMENT_CLASS = Document

    @staticmethod
    def __new__(cls, desc=None, *args, backend=None, **kwargs):
        if cls is not Collection:
            return super(Collection, cls).__new__(desc, *args, **kwargs)

        if isinstance(desc, str):
            desc = urisplit(desc)
        elif not isinstance(desc, AttributeTree):
            desc = AttributeTree(desc)

        plugin_name = backend or desc.schema or ""

        if plugin_name is not None:
            n_cls = find_plugin(f"{plugin_name}://",
                                pattern=f"{__package__}.plugins.Plugin{{name}}",
                                fragment="Collection")
        else:
            n_cls = cls
        return object.__new__(n_cls)

    def __init__(self, desc=None,
                 *args,
                 mode="rw",
                 id_hasher=None,
                 handler=None,
                 request_proxy=None,
                 **kwargs):
        super().__init__()

        if isinstance(desc, str):
            self._envs = urisplit(desc)
        elif not isinstance(desc, AttributeTree):
            self._envs = AttributeTree(desc)
        else:
            self._envs = desc
        logger.debug(f"Open {self.__class__.__name__} : {self._envs.schema}://{self._envs.authorize}/{self._envs.path}")

        self._mode = mode
        self._id_hasher = id_hasher or "{_id}"

        if request_proxy is not None:
            self._handler = request_proxy(handler)
        else:
            self._handler = handler

    @property
    def envs(self):
        return self._envs

    @property
    def mode(self):
        return self._mode

    @property
    def is_writable(self):
        return "w" in self._mode

    @property
    def handler(self):
        return self._handler

    # mode in ["", auto_inc  , glob ]
    def guess_id(self, d, auto_inc=True):
        fid = None
        if callable(self._id_hasher):
            fid = self._id_hasher(self, d, auto_inc)
        elif isinstance(self._id_hasher, str):
            try:
                fid = self._id_hasher.format_map(d)
            except KeyError:
                if auto_inc:
                    fid = self._id_hasher.format(_id=self.count())
                else:
                    raise RuntimeError(f"Can not get id from {d}!")

        return fid

    def open_document(self, fid, mode):
        logger.debug(f"Opend Document: {fid} mode=\"{mode}\"")
        raise NotImplementedError()

    def insert(self, *args, **kwargs):
        return self.insert_one(*args, **kwargs)

    def open(self, *args, mode="rw", **kwargs):
        if "w" in mode and self.is_writable:
            return self.insert_one(*args, **kwargs)
        elif "w" not in mode:
            return self.find_one(*args, **kwargs)
        else:
            raise RuntimeWarning("Collection is not writable!")

    def find_one(self, predicate=None, *args, **kwargs):
        raise NotImplementedError()

    def find(self, predicate=None, projection=None, *args, **kwargs):
        raise NotImplementedError()

    def insert_one(self, data=None, *args, **kwargs) -> InsertOneResult:
        raise NotImplementedError()

    def insert_many(self, documents, *args, **kwargs) -> InsertManyResult:
        return [self.insert_one(doc, *args, **kwargs) for doc in documents]

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


class FileCollection(Collection):

    def __init__(self, uri, *args,
                 file_extension=".dat",
                 file_factory=None,
                 **kwargs):

        super().__init__(uri, *args, **kwargs)

        if isinstance(uri, str):
            uri = urisplit(uri)

        path = getattr(uri, "path", ".")

        if isinstance(path, str) and not path.endswith(file_extension):
            path = f"{path}{{_id}}{file_extension}"

        self._path = pathlib.Path(path).resolve().expanduser()

        logger.debug(self._path)

        if self._path.suffix == '':
            self._path = self._path.with_suffix(file_extension)

        if "{_id}" not in self._path.stem:
            self._path = self._path.with_name(f"{self._path.stem}{{_id}}{self._path.suffix}")

        if not self._path.parent.exists():
            if "w" not in self._mode:
                raise RuntimeError(f"Can not make dir {self._path}")
            else:
                self._path.parent.mkdir()
        elif not self._path.parent.is_dir():
            raise NotADirectoryError(self._path.parent)

        self._file_factory = file_factory

    def guess_id(self, d, auto_inc=True):
        fid = super().guess_id(d, auto_inc=auto_inc)

        if fid is None and auto_inc:
            fid = self.count()

        return fid

    def guess_filepath(self, *args, **kwargs):
        return self._path.with_name(self._path.name.format(_id=self.guess_id(*args, **kwargs)))

    def open_document(self, fid, mode=None):
        fpath = self.guess_filepath({"_id": fid})
        logger.debug(f"Opend Document: {fpath} mode=\"{ mode or self.mode}\"")
        return Document(root=self._file_factory(fpath, mode or self.mode), envs=self.envs, handler=self._handler)

    def insert_one(self, data=None, *args,  **kwargs):
        doc = self.open_document(self.guess_id(data or kwargs, auto_inc=True), mode="w")
        doc.update(data or kwargs)
        return doc

    def find_one(self, predicate=None, projection=None, **kwargs):
        fpath = self.guess_filepath(predicate or kwargs)

        doc = None
        if fpath.exists():
            doc = self.open_document(fpath)
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

    def count(self, predicate=None,   *args, **kwargs) -> int:

        if predicate is None:
            logger.warning("NOT IMPLEMENTED! count by predicate")

        return len(list(self._path.parent.glob(self._path.name.format(_id="*"))))
