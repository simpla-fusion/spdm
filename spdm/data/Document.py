
from spdm.util.LazyProxy import LazyProxy


class Document(object):
    def __init__(self, *args, schema=None, **kwargs):
        self._schema = schema

    @property
    def root(self):
        return NotImplemented

    @property
    def entry(self):
        return LazyProxy(self.root)

    @property
    def schema(self):
        return self._schema

    def valid(self, schema=None):
        return True

    def check(self, predication, **kwargs) -> bool:
        raise NotImplementedError()

    def fetch(self, projection, **kwargs):
        raise NotImplementedError()

    def update(self,  data, **kwargs):
        raise NotImplementedError()

    def put(self, path, value):
        raise NotImplementedError()

    def get(self, path):
        raise NotImplementedError()

    def remove(self, path):
        raise NotImplementedError()

    def __setitem__(self, path, value):
        return self.put(path, value)

    def __getitem__(self, path):
        return self.get(path)

    def __delitem__(self, path):
        return self.remove(path)
