import collections
import pathlib
from functools import cached_property
from typing import Any, Dict, List, NewType, Tuple

from ..util.logger import logger
from ..util.uri_utils import uri_split, uri_merge
from .Document import Document
from .File import File
from .SpObject import SpObject

InsertOneResult = collections.namedtuple("InsertOneResult", "inserted_id success")
InsertManyResult = collections.namedtuple("InsertManyResult", "inserted_ids success")
UpdateResult = collections.namedtuple("UpdateResult", "inserted_id success")
DeleteResult = collections.namedtuple("DeleteResult", "deleted_id success")


class Collection(SpObject):
    ''' Collection of documents
    '''

    def __init__(self, *args, mode="rw", envs=None,   **kwargs):
        super().__init__()

        self._mode = mode
        self._envs = collections.ChainMap(kwargs, envs or {})
        logger.info(f"Create {self.__class__.__name__} : {self._metadata}")

    def __del__(self):
        logger.info(f"Close  {self.__class__.__name__} : {self._metadata}")

    @property
    def schema(self):
        return self._metadata.get("schema", None)

    @cached_property
    def envs(self):
        return self._envs

    @property
    def mode(self):
        return self._mode

    @property
    def is_writable(self):
        return "w" in self.mode or 'x' in self.mode

    def guess_id(self, *args, **kwargs):
        return NotImplemented

    @property
    def next_id(self):
        raise NotImplementedError()

    # def open(self, *args, mode=None, **kwargs):
    #     mode = mode or self.mode
    #     if "x" in mode:
    #         return self.insert_one(*args,  **kwargs)
    #     else:
    #         return self.find_one(*args,   **kwargs)

    def create(self, *args, **kwargs):
        return self.insert_one(*args, mode="x", **kwargs)

    def insert(self, data, *args, **kwargs):
        if isinstance(data, list):
            return self.insert_many(data, *args, **kwargs)
        else:
            return self.insert_one(data, *args, **kwargs)

    def insert_one(self, data, * args,  **kwargs) -> InsertOneResult:
        doc = Document(self.guess_id(*args, **kwargs) or self.next_id, **kwargs)
        if data is not None:
            doc.update(data)
        return doc

    def insert_many(self, documents, *args, **kwargs) -> InsertManyResult:
        return [self.insert_one(doc, *args, **kwargs) for doc in documents]

    def find_one(self, *args, predicate=None, **kwargs):
        return Document(self.guess_id(*args, **kwargs), **kwargs).fetch(predicate)

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
