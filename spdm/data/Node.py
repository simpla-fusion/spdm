from spdm.util.LazyProxy import LazyProxy


class Holder:
    pass


class Handler(LazyProxy.Handler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def iter(self, holder, *args, **kwargs):
        raise StopIteration()


class Node(object):
    def __init__(self, holder=None, *args,  handler=None, **kwargs):
        super().__init__()
        self._holder = holder
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
