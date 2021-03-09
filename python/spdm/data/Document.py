import collections
from typing import Any, Dict, List
from ..util.logger import logger
from ..util.LazyProxy import LazyProxy
from .DataObject import DataObject
from .Entry import Entry


class Document(DataObject):

    def __init__(self,  path=None, *args, fid=None, mode="r", envs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = path
        self._mode = mode
        self._fid = fid
        self._data = None
        self._envs = collections.ChainMap(envs or {}, kwargs)

    def __del__(self):
        self.close()

    def copy(self, other):
        if isinstance(other, Document):
            return self.root.copy(other.root)
        else:
            return self.root.copy(other)

    @property
    def root(self):
        return Entry(self._data)

    @property
    def entry(self):
        return self.root.lazy_entry

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
    def mode(self):
        return self._mode

    def validate(self, schema=None):
        raise NotImplementedError()

    def close(self):
        self._data = None

    def flush(self):
        raise NotImplementedError()

    def check(self, predication, **kwargs) -> bool:
        raise NotImplementedError()

    def update(self, d: Dict[str, Any]):
        self.root.update(d)

    def fetch(self, proj: Dict[str, Any] = None):
        raise NotImplementedError()

    def dump(self):
        raise NotImplementedError()
