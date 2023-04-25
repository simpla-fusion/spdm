import collections
from typing import Any, Dict, List, TypeVar

from ..utils.logger import logger
from .Dict import Dict
from .Entry import Entry

_TDocument = TypeVar("_TDocument", bound="Document")


class Document(Entry):

    def __init__(self,  *args,  fid=None, envs=None, schema=None, **kwargs):

        super(Entry, self).__init__(*args, **kwargs)

        # self._fid = fid
        # self._path = path
        # self._mode = mode
        # self._envs = collections.ChainMap(envs or {}, kwargs)
        # self._parent = parent
        # self._schema = schema

    def __repr__(self) -> str:
        return Dict.__repr__(self)

    @property
    def schema(self):
        return self._schema

    @property
    def envs(self):
        return self._envs

    @property
    def fid(self):
        return self._fid
