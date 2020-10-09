import collections
from ..util.LazyProxy import LazyProxy


class Cursor(collections.Iterable):
    def __init__(self, cursor, **kwargs):
        self._cursor = cursor
        return

    def __iter__(self):
        return self

    def __next__(self):
        return LazyProxy(next(self._cursor), [], lambda s, o, p: o[p])
