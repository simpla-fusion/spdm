import collections
import re
import urllib
from pathlib import Path
from typing import Any, Dict, List, NewType, Tuple

import numpy

from spdm.util.logger import logger
from spdm.util.utilities import whoami
from spdm.util.LazyProxy import LazyProxy
from .Document import Document

InsertOneResult = collections.namedtuple("InsertOneResult", "inserted_id success")
InsertManyResult = collections.namedtuple("InsertManyResult", "inserted_ids success")
UpdateResult = collections.namedtuple("UpdateResult", "upserted_id success")
DeleteResult = collections.namedtuple("DeleteResult", "deleted_id success")


class Collection(object):
    ''' Collection of document

        API is compatible with MongoDB
    '''

    def __init__(self, *args,  **kwargs):
        super().__init__()

    def create(self, *args, **kwargs):
        return LazyProxy(self.insert_one(*args, **kwargs))

    def open(self, *args, **kwargs):
        return LazyProxy(self.find_one(*args, **kwargs))

    def find_one(self, predicate=None, *args, **kwargs):
        raise NotImplementedError(whoami(self))

    def find(self, predicate: Document = None, projection: Document = None, *args, **kwargs):
        raise NotImplementedError(whoami(self))

    def insert_one(self, document: Document, *args, **kwargs) -> InsertOneResult:
        raise NotImplementedError(whoami(self))

    def insert_many(self, documents, *args, **kwargs) -> InsertManyResult:
        return [self.insert_one(doc, *args, **kwargs) for doc in documents]

    def replace_one(self, predicate: Document, replacement: Document,  *args, **kwargs) -> UpdateResult:
        raise NotImplementedError(whoami(self))

    def update_one(self, predicate: Document, update: Document,  *args, **kwargs) -> UpdateResult:
        raise NotImplementedError(whoami(self))

    def update_many(self, predicate: Document, updates: list,  *args, **kwargs) -> UpdateResult:
        return [self.update_one(predicate, update, *args, **kwargs) for update in updates]

    def delete_one(self, predicate: Document,  *args, **kwargs) -> DeleteResult:
        raise NotImplementedError(whoami(self))

    def delete_many(self, predicate: Document, *args, **kwargs) -> DeleteResult:
        raise NotImplementedError(whoami(self))

    def count(self, predicate: Document = None,   *args, **kwargs) -> int:
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
