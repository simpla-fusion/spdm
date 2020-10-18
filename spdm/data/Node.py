from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger


class Node(object):
    def __init__(self, holder=None, *args, prefix=None, envs=None,  **kwargs):
        super().__init__()
        self._holder = holder
        self._prefix = prefix or []
        self._envs = envs or {}
        logger.debug((self._prefix, self._envs))

    @property
    def holder(self):
        return self._holder

    @property
    def prefix(self):
        return self._prefix

    @property
    def envs(self):
        return self._envs

    @property
    def entry(self):
        return LazyProxy(self, handler=self.__class__)

    def put(self, p, v):
        raise NotImplementedError()

    def get(self, p):
        raise NotImplementedError()

    def get_value(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def iter(self, *args, **kwargs):
        raise NotImplementedError()
