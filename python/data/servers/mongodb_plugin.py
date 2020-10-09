import collections
import pymongo

from spdm.data.collection import (Collection, DObject, InsertOneResult,
                                  UpdateResult)
from spdm.data.data_entry import DataEntry
from spdm.util.logger import logger
from spdm.data.converter import cannonical_data

__plugin_spec__ = {
    "name": "mongodb",
    "url_pattern": ["mongodb://*"],
}


# class MongoDBEntry(DataEntry):
#     """Entry of MongoDB document."""
#     pass


class MongoDBCollection(Collection):
    """Wrapper of Mongodb collection."""

    def __init__(self, collection, **kwargs):
        self._collection = collection

    @property
    def collection(self):
        return self._collection

    def find_one(self, *args,  **kwargs):
        return self._collection.find_one(*args, **kwargs)

    def find(self,  predicate: DObject = None, projection: DObject = None,
             *args, **kwargs):
        for res in self._collection.find(predicate, projection,
                                         *args, **kwargs):
            if isinstance(res, collections.abc.Mapping):
                yield DataEntry(res)
            else:
                yield res

    def insert_one(self, doc, *args, **kwargs):
        return self._collection.insert_one(cannonical_data(doc), *args, **kwargs).inserted_id

    def insert_many(self, documents,
                    *args, **kwargs):
        return self._collection.insert_many(cannonical_data(documents), *args, **kwargs)

    def update_one(self, *args, **kwargs):
        return self._collection.update_one(*args, **kwargs)

    def update_many(self, *args, **kwargs):
        return self._collection.update_many(*args, **kwargs)

    def replace_one(self, *args, **kwargs):
        return self._collection.replace_one(*args, **kwargs)

    def delete_one(self, *args, **kwargs):
        return self._collection.delete_one(*args, **kwargs)

    def delete_many(self, *args, **kwargs):
        return self._collection.delete_many(*args, **kwargs)


class MongoDBConnect(object):

    __plugin_spec__ = {"name": "mongodb",
                       "url_pattern": ["mongodb://*"]}

    def __init__(self, netloc, prefix=None, scheme=None, *args,  **kwargs):
        super().__init__()
        logger.info(
            f"Open connection MongoDB : {scheme or 'mongodb'}://{netloc}/")
        self._client = pymongo.MongoClient(
            f"{scheme or 'mongodb'}://{netloc}/")

        self._database = self._client[str(prefix).strip("/").replace("/", "_")]
        # self._collection = self._database[prefix[1]]

    @classmethod
    def connect(cls, url, *args, **kwargs):
        return MongoDBConnect(url, *args, **kwargs)

    def open(self, path, *args, **kwargs):
        return MongoDBCollection(self._database[str(path).strip("/")
                                                .replace("/", "_")])


__SP_EXPORT__ = MongoDBConnect
