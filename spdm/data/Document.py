from typing import Any, Dict, List
import collections
from spdm.util.logger import logger
from spdm.util.AttributeTree import AttributeTree

from .Node import Node
from .DataObject import DataObject


class Document(object):

    def __init__(self, desc, *args,  request_proxy=None, parent=None, **kwargs):
        super().__init__()
        if isinstance(desc, DataObject):
            self._holder = desc
            self._desc = AttributeTree(schema=self._holder.__class__.__name__)
        else:
            self._desc = AttributeTree(collections.ChainMap(desc or {},   kwargs))
            self._holder = DataObject(desc,*args, **kwargs)

        self._parent = parent
        self._request_proxy = request_proxy
        logger.debug(f"Opend Document type='{self._holder.__class__.__name__}' fid={self._desc.fid} ")

    @property
    def holder(self):
        return self._holder

    @property
    def root(self):
        r_node = getattr(self._holder, "root", self._holder)
        if isinstance(r_node, Node) and not self._request_proxy:
            return r_node
        else:
            return Node(r_node,  request_proxy=self._request_proxy)

    @property
    def description(self):
        return self._desc

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
