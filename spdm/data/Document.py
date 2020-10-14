
from spdm.util.LazyProxy import LazyProxy
from typing import (Dict, List, Any)


class Document(object):
    def __init__(self, *args, schema=None, root=None, handler=None, collection=None, **kwargs):
        print(f"Init  {self.__class__.__name__} {root}")

        self._schema = schema
        self._root = root
        self._handler = handler
        self._collection = collection

    @property
    def root(self):
        return self._root

    @property
    def entry(self):
        return LazyProxy(self.root, handler=self._handler)

    @property
    def schema(self):
        return self._schema

    def valid(self, schema=None):
        return True

    def check(self, predication, **kwargs) -> bool:
        raise NotImplementedError()

    def put(self, p, v):
        return self._handler.put(self.root, p, v)

    def get(self, p):
        return self._handler.get(self.root, p)

    def iter(self, p):
        for obj in self._handler.iter(self.root, p):
            yield obj

    def update(self, d: Dict[str, Any]):
        return self._handler.put(self.root, [], d)

    def fetch(self, proj: Dict[str, Any] = None):
        return self._handler.get(self.root, proj)

    def dump(self):
        return self._handler.dump(self.root)
