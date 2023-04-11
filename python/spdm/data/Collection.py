import collections
import collections.abc
import typing

from ..util.sp_export import sp_find_module
from ..util.uri_utils import URITuple, uri_split
from .Connection import Connection
from .Entry import Entry
from .Mapper import Mapper

InsertOneResult = collections.namedtuple("InsertOneResult", "inserted_id success")
InsertManyResult = collections.namedtuple("InsertManyResult", "inserted_ids success")
UpdateResult = collections.namedtuple("UpdateResult", "inserted_id success")
DeleteResult = collections.namedtuple("DeleteResult", "deleted_id success")


class Collection(Connection):
    ''' Collection of documents
    '''
    _registry = {}
    _plugin_prefix = "spdm.plugins.data.Plugin"

    # @classmethod
    # def register(cls, name: typing.Union[str, typing.List[str]], other_cls=None):
    #     """
    #     Decorator to register a class to the registry.
    #     """
    #     if other_cls is not None:
    #         if isinstance(name, str):
    #             cls._registry[name] = other_cls
    #         elif isinstance(name, collections.abc.Sequence):
    #             for n in name:
    #                 cls._registry[n] = other_cls

    #         return other_cls
    #     else:
    #         def decorator(o_cls):
    #             Collection.register(name, o_cls)
    #             return o_cls
    #         return decorator

    # @classmethod
    # def create(cls, path, *args, **kwargs):
    #     n_cls_name = None

    #     if "protocol" in kwargs:
    #         protocol = kwargs.get("protocol")
    #         n_cls_name = protocol
    #     elif isinstance(path, collections.abc.Mapping):
    #         n_cls_name = path.get("$class", None)
    #     elif isinstance(path, (str, URITuple)):
    #         uri = uri_split(path)
    #         n_cls_name = uri.protocol

    #     return super().create(n_cls_name, path, *args, **kwargs)

    @classmethod
    def _guess_class_name(cls, path, *args,  **kwargs) -> typing.Optional[str]:
        n_cls_name = None

        if "protocol" in kwargs:
            protocol = kwargs.get("protocol")
            n_cls_name = protocol
        elif isinstance(path, collections.abc.Mapping):
            n_cls_name = path.get("$class", None)
        elif isinstance(path, (str, URITuple)):
            uri = uri_split(path)
            n_cls_name = uri.protocol

        return n_cls_name

    def __new__(cls,  *args, **kwargs):
        if cls is not Collection:
            return object.__new__(cls)
        else:
            return super().__new__(cls, *args, **kwargs)

    def __init__(self, uri, *args,  mapper: typing.Optional[Mapper] = None,   **kwargs):
        super().__init__(uri, *args, **kwargs)
        self._mapper = mapper

    @property
    def mapper(self) -> typing.Optional[Mapper]:
        return self._mapper

    def guess_id(self, predicate, *args, fragment: int = None, **kwargs) -> int:
        if isinstance(predicate, int):
            return predicate
        elif isinstance(predicate, str):
            return int(predicate)
        elif fragment is not None:
            return int(fragment) if isinstance(fragment, str) else fragment
        else:
            raise NotImplementedError(f"predicate={predicate} fragment={fragment}")

    @property
    def next_id(self):
        raise NotImplementedError()

    def _mapping(self, entry: Entry) -> Entry:
        return self.mapper.map(entry) if self.mapper is not None else entry

    def create_one(self, *args, **kwargs):
        return self.insert_one(*args, mode="x", **kwargs)

    def create_many(self, docs: typing.List[typing.Any], *args, **kwargs):
        return [self.create_one(doc, *args, mode="x", **kwargs) for doc in docs]

    def create_doc(self, docs, *args, **kwargs):
        if isinstance(docs, collections.abc.Sequence):
            return self.create_many(docs, *args, **kwargs)
        else:
            return self.create_one(docs, *args, **kwargs)

    def insert_one(self, doc, * args,  **kwargs) -> InsertOneResult:
        raise NotImplementedError()

    def insert_many(self, docs: typing. List[typing.Any], *args, **kwargs) -> InsertManyResult:
        return [self.insert_one(doc, *args, **kwargs) for doc in docs]

    def insert(self, docs, *args, **kwargs):
        if isinstance(docs, collections.abc.Sequence):
            return self.insert_many(docs, *args, **kwargs)
        else:
            return self.insert_one(docs, *args, **kwargs)

    def find_one(self, *args, **kwargs) -> Entry:
        raise NotImplementedError()

    def find_many(self, *args, **kwargs) -> typing.List[Entry]:
        raise NotImplementedError()

    def find(self, predicate, projection=None, only_one=False, **kwargs) -> typing.Union[Entry,  typing.List[Entry]]:
        # if not isinstance(predicate, str) and isinstance(predicate, collections.abc.Sequence):
        if not only_one:
            return self.find_many(predicate, projection,  **kwargs)
        else:
            return self.find_one(predicate, projection,  **kwargs)

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

    def create_indexes(self, indexes: typing.List[str], session=None, **kwargs):
        raise NotImplementedError()

    def create_index(self, keys: typing.List[str], session=None, **kwargs):
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
