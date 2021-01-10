import collections
from typing import Any, Dict, List
import pathlib
from ..util.AttributeTree import AttributeTree
from ..util.logger import logger
from .Node import Node
from .DataObject import DataObject

import pathlib


class Document(DataObject):

    def __init__(self, *args, fid=None, mode="r", path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = mode
        self._fid = fid
        self._path = pathlib.Path(path or self.metadata.path or pathlib.Path.cwd())
        self._data = None

    def __del__(self):
        pass

    def copy(self, other):
        if isinstance(other, Document):
            return self.root.copy(other.root)
        else:
            return self.root.copy(other)

    @property
    def root(self):
        return Node(self._data)

    @property
    def entry(self):
        return self.root.entry

    @property
    def path(self):
        return self._path

    @property
    def fid(self):
        return self._fid

    @property
    def mode(self):
        return self._mode

    def validate(self, schema=None):
        raise NotImplementedError()

    def flush(self):
        raise NotImplementedError()

    def check(self, predication, **kwargs) -> bool:
        raise NotImplementedError()

    def update(self, d: Dict[str, Any]):
        self._data = AttributeTree(d) if not isinstance(d, AttributeTree) else d
        # raise NotImplementedError()

    def fetch(self, proj: Dict[str, Any] = None):
        raise NotImplementedError()

    def dump(self):
        raise NotImplementedError()
