from typing import Any, Dict, List

from spdm.util.logger import logger

from .Node import Node
from .Plugin import find_plugin


class Document(Node):

    @staticmethod
    def __new__(cls, desc=None, *args, format_type=None,  **kwargs):
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

    def __init__(self, desc=None, *args, fid=None, root=None, collection=None, schema=None, mode="rw", **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug(f"Opend {self.__class__.__name__} fid={fid} ")
        self._fid = fid
        self._schema = schema
        self._collection = collection
        self._root = root if root is not None else Node(root or {})
        self._mode = mode

    @property
    def root(self):
        return self._root

    @property
    def entry(self):
        return self.root.entry

    @property
    def schema(self):
        return self._schema

    @property
    def mode(self):
        return self._mode

    def copy(self, other):
        if isinstance(other, Document):
            return self.root.copy(other.root)
        else:
            return self.root.copy(other)

    def load(self, *args, **kwargs):
        raise NotImplementedError()

    def save(self,  *args, **kwargs):
        raise NotImplementedError()

    def valid(self, schema=None):
        return True

    def flush(self):  # dump data from memory to storage (file or database)
        raise NotImplementedError()

    def check(self, predication, **kwargs) -> bool:
        raise NotImplementedError()

    def update(self, d: Dict[str, Any]):
        return self.root.put([], d)

    def fetch(self, proj: Dict[str, Any] = None):
        return self.root.get(proj)

    def dump(self):
        return self.root.dump()
