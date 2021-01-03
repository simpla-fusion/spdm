import collections
from typing import Any, Dict, List

from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger

from .DataObject import DataObject
from .Node import Node


class Document(DataObject):

    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)

   
    def copy(self, other):
        if isinstance(other, Document):
            return self.root.copy(other.root)
        else:
            return self.root.copy(other)

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
