import collections
from typing import Any, Dict, List

from ..util.AttributeTree import AttributeTree
from ..util.logger import logger
from .Node import Node
from .DataObject import DataObject


class Document(DataObject):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def copy(self, other):
        if isinstance(other, Document):
            return self.root.copy(other.root)
        else:
            return self.root.copy(other)

    @property
    def root(self):
        return Node(self)

    @property
    def entry(self):
        return self.root.entry

    def validate(self, schema=None):
        raise NotImplementedError()

    def flush(self):
        raise NotImplementedError()

    def check(self, predication, **kwargs) -> bool:
        raise NotImplementedError()

    def update(self, d: Dict[str, Any]):
        raise NotImplementedError()

    def fetch(self, proj: Dict[str, Any] = None):
        raise NotImplementedError()

    def dump(self):
        raise NotImplementedError()
