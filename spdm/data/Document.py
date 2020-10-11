
from .Entry import Entry
from spdm.util.LazyProxy import LazyProxy


class Document(object):
    def __init__(self, *args, collection=None, **kwargs):
        self._collection = collection

    @property
    def root(self):
        return self._root

    @property
    def schema(self):
        return self._schema

    def valid(self, schema=None):
        return self.m_root_

    def fetch(self, path, **kwargs):
        return None

    def update(self,  path, **kwargs):
        return None
