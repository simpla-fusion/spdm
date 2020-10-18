from typing import Any, Dict, List

from spdm.util.logger import logger

from .Node import Node
from .Plugin import find_plugin


class Document(Node):

    @staticmethod
    def __new__(cls, desc=None, *args, format_type=None, **kwargs):
        if cls is not Document or isinstance(desc, Node):
            return super(Document, cls).__new__(cls)

        if format_type is not None:
            desc = f"{format_type}://"

        if desc is None:
            return super(Document, cls).__new__(cls)
        else:
            n_cls = find_plugin(desc,
                                pattern=f"{__package__}.plugins.Plugin{{name}}",
                                fragment="Document")
            return object.__new__(n_cls)

    def __init__(self, root=None, *args, collection=None, schema=None, **kwargs):
        logger.debug(f"Opend {self.__class__.__name__}")

        super().__init__(*args, **kwargs)
        self._schema = schema
        self._collection = collection
        self._root = root if isinstance(root, Node) else Node(root or {})

    @property
    def root(self):
        return self._root

    @property
    def entry(self):
        return self.root.entry

    @property
    def schema(self):
        return self._schema

    def valid(self, schema=None):
        return True

    def flush(self):  # dump data from memory to storage (file or database)
        raise NotImplementedError()

    def check(self, predication, **kwargs) -> bool:
        raise NotImplementedError()

    def update(self, d: Dict[str, Any]):
        return self._handler.put(self.root, [], d)

    def fetch(self, proj: Dict[str, Any] = None):
        return self._handler.get(self.root, proj)

    def dump(self):
        return self._handler.dump(self.root)
