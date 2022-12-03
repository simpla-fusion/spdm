import collections
import pathlib
from functools import cached_property
from typing import Any,  NewType, Tuple, Union

from ..common.tags import _not_found_, _undefined_
from ..util.logger import logger
from ..util.uri_utils import URITuple, uri_merge, uri_split
from .Connection import Connection
from .Entry import Entry
from .File import File
from .Mapper import Mapper
from .SpObject import SpObject
from .List import List
InsertOneResult = collections.namedtuple("InsertOneResult", "inserted_id success")
InsertManyResult = collections.namedtuple("InsertManyResult", "inserted_ids success")
UpdateResult = collections.namedtuple("UpdateResult", "inserted_id success")
DeleteResult = collections.namedtuple("DeleteResult", "deleted_id success")


class Collection(Connection):
    ''' Collection of documents
    '''
    def __new__(cls, path, *args, **kwargs):
        if cls is not Collection:
            return object.__new__(cls)

        if "protocol" in kwargs:
            protocol = kwargs.get("protocol")
            n_cls_name = f".{protocol.lower()}"
        elif isinstance(path, collections.abc.Mapping):
            n_cls_name = path.get("$class", None)
        elif isinstance(path, (str, URITuple)):
            uri = uri_split(path)
            n_cls_name = f".{uri.protocol.lower()}"

        return Collection.object_new(n_cls_name)

    def __init__(self, uri, *args,  mapper: Mapper = _undefined_,   **kwargs):
        super().__init__(uri, *args, **kwargs)
        self._mapper = mapper

    @property
    def mapper(self) -> Mapper:
        return self._mapper

    def guess_id(self, *args, **kwargs):
        return NotImplemented

    @property
    def next_id(self):
        raise NotImplementedError()

    def _mapping(self, entry: Entry) -> Entry:
        return self.mapper.map(entry) if self.mapper is not None else entry

    def create_one(self, *args, **kwargs):
        return self.insert_one(*args, mode="x", **kwargs)

    def create_many(self, docs: List[Any], *args, **kwargs):
        return [self.create_one(doc, *args, mode="x", **kwargs) for doc in docs]

    def create(self, docs, *args, **kwargs):
        if isinstance(docs, collections.abc.Sequence):
            return self.create_many(docs, *args, **kwargs)
        else:
            return self.create_one(docs, *args, **kwargs)

    def insert_one(self, doc, * args,  **kwargs) -> InsertOneResult:
        raise NotImplementedError()

    def insert_many(self, docs: List[Any], *args, **kwargs) -> InsertManyResult:
        return [self.insert_one(doc, *args, **kwargs) for doc in docs]

    def insert(self, docs, *args, **kwargs):
        if isinstance(docs, collections.abc.Sequence):
            return self.insert_many(docs, *args, **kwargs)
        else:
            return self.insert_one(docs, *args, **kwargs)

    def find_one(self, *args, **kwargs) -> Entry:
        res = self.find(*args, only_one=True, **kwargs)
        return res[0] if len(res) > 0 else None

    def find_many(self, *args, **kwargs) -> List[Entry]:
        res = self.find(*args, only_one=True, **kwargs)
        return res[0] if len(res) > 0 else None

    def find(self, predicate=None, projection=None, *args, **kwargs) -> Union[Entry, List[Entry]]:
        if not isinstance(predicate, str) and isinstance(predicate, collections.abc.Sequence):
            return self.find_many(predicate=None, projection=None, *args, **kwargs)
        else:
            return self.find_one(predicate=None, projection=None, *args, **kwargs)

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
