from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger


class Node(object):
    def __init__(self, holder=None, *args, prefix=None, envs=None,  **kwargs):
        super().__init__()
        self._holder = holder
        self._prefix = prefix or []
        self._envs = envs or {}

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

    def put(self, path, v, *args, **kwargs):
        raise NotImplementedError(self.__class__.__name__)

    def get(self, path, *args, **kwargs):
        raise NotImplementedError(self.__class__.__name__)

    def get_value(self, path, *args, **kwargs):
        return self.get(path, *args, **kwargs)

    def iter(self, path,  *args, **kwargs):
        raise NotImplementedError(self.__class__.__name__)
