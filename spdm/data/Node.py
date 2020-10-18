from spdm.util.LazyProxy import LazyProxy


class Holder:
    pass


class Holder:
    def __init__(self, d, *args, **kwargs):
        self._data = d

    @property
    def data(self):
        return self._data


class Handler(LazyProxy.Handler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def iter(self, holder, *args, **kwargs):
        raise StopIteration()


class Node(object):
    def __init__(self, holder=None, *args,  handler=None, **kwargs):
        super().__init__()
        self._holder = holder if isinstance(holder, Holder) else Holder(holder)
        self._handler = handler

    @property
    def holder(self):
        return self._holder

    @property
    def handler(self):
        return self._handler

    @property
    def entry(self):
        return LazyProxy(self._holder, handler=self.handler)

    def put(self, p, v):
        return self._handler.put(self._holder, p, v)

    def get(self, p):
        return self._handler.get(self._holder, p)

    def get_value(self, *args, **kwargs):
        return self._handler.get_value(self._holder, *args, **kwargs)

    def iter(self, *args, **kwargs):
        return self._handler.iter(self._holder, *args, **kwargs)
