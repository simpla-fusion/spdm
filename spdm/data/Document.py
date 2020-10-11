
from spdm.util.LazyProxy import LazyProxy
from typing import (Dict, List, Any)


class Document(object):
    def __init__(self, *args, schema=None, root=None, handler=None, **kwargs):
        self._schema = schema
        self._root = root
        self._handler = handler

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

    def update(self, d: Dict[str, Any]):
        return self._handler.put(self.root, [], d)

    def fetch(self, proj: Dict[str, Any] = None):
        return self._handler.get(self.root, proj)

    def dump(self):
        return self._handler.dump(self.root)
