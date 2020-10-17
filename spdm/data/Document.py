
from typing import (Dict, List, Any)

from .Node import Node


class Document(Node):

    @staticmethod
    def __new__(cls, desc, *args, **kwargs):
        if cls is not Document:
            return super(Document, cls).__new__(desc, *args, **kwargs)

        return cls

    def __init__(self, *args, collection=None, schema=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._schema = schema
        self._collection = collection

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
