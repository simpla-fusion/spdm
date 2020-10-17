from spdm.util.LazyProxy import LazyProxy


class Handler(LazyProxy.Handler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Node(object):
    def __init__(self,   *args, holder=None, handler=None, **kwargs):
        super().__init__(*args, **kwargs)
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
