import collections
from typing import Any, Dict, List

from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger

from .DataObject import DataObject
from .Node import Node


class Document(DataObject):

    def __init__(self, data=None,  *args,  **kwargs):
        self._holder = None
        super().__init__(data, *args,   **kwargs)
        
        logger.debug(
            f"Opend Document type='{self.__class__.__name__}' path='{self.metadata.oid or self.metadata.path or self.metadata['$schema'].path}' ")

    @property
    def holder(self):
        return self._holder

    @property
    def root(self):
        r_node = getattr(self._holder, "root", self._holder)
        if isinstance(r_node, Node):
            return r_node
        else:
            return Node(r_node)

    @property
    def entry(self):
        return self.root.entry

    def copy(self, other):
        if isinstance(other, Document):
            return self.root.copy(other.root)
        else:
            return self.root.copy(other)

    def load(self, *args, **kwargs):
        self._holder = DataObject(*args, **kwargs)
        self._desc = AttributeTree(schema=self._holder.__class__.__name__)

    def save(self,  *args, **kwargs):
        return self._holder.save(*args, **kwargs)

    def validate(self, schema=None):
        return self._holder.validate(schema)

    def flush(self):
        return self._holder.flush()

    def check(self, predication, **kwargs) -> bool:
        return self._holder.check(predication, **kwargs)

    def update(self, d: Dict[str, Any]):
        return self._holder.put([], d)

    def fetch(self, proj: Dict[str, Any] = None):
        return self._holder.get(proj)

    def dump(self):
        return self._holder.dump()
