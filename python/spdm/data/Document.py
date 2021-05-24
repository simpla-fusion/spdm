import collections
from typing import Any, Dict, List
from functools import cached_property
from spdm.data.AttributeTree import AttributeTree
from ..util.logger import logger
from .DataObject import DataObject
from .Entry import Entry


class Document(DataObject):

    def __init__(self,  *args,  fid=None, parent=None, path=None, mode="r", envs=None, schema=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._fid = fid
        self._path = path
        self._mode = mode
        self._envs = collections.ChainMap(envs or {}, kwargs)
        self._parent = parent
        self._schema = schema

    def __del__(self):
        pass

    def copy(self, other):
        if isinstance(other, Document):
            return self.entry.copy(other.entry)
        else:
            return self.entry.copy(other)

    @cached_property
    def entry(self):
        return Entry(self)

    @cached_property
    def attributes(self):
        return AttributeTree(self.entry)

    @property
    def schema(self):
        return self._schema

    @property
    def envs(self):
        return self._envs

    @property
    def path(self):
        return self._path

    @property
    def fid(self):
        return self._fid

    @property
    def parent(self):
        return self._parent

    @property
    def mode(self):
        return self._mode

    def validate(self, schema=None):
        raise NotImplementedError()

    def flush(self):
        raise NotImplementedError()

    def check(self, predication, **kwargs) -> bool:
        raise NotImplementedError()

    def update(self, d: Dict[str, Any], *args, **kwargs):
        self.entry.update(d)

    def fetch(self, proj: Dict[str, Any] = None, *args, **kwargs):
        if proj is not None:
            raise NotImplementedError()
        else:
            return self

    def dump(self):
        raise NotImplementedError()
