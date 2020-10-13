
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
import pathlib
import collections
from xml.etree import (ElementTree, ElementInclude)
import numpy as np

Linker = collections.namedtuple("Linker", "schema path")


class Handler(LazyProxy.Handler):
    DELIMITER = LazyProxy.DELIMITER

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HandlerProxy(Handler):
    def __init__(self, mapper, *args, target=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._mapper = mapper
        self._target = target

    @property
    def target(self):
        return self._target

    def set_target(self, t):
        self._target = t

    def get(self, grp, path=[], *args, **kwargs):
        obj = self._mapper.get(path)

        if obj is None:
            return self._target.get(grp, path, *args, **kwargs)
        elif isinstance(obj, Linker):
            return self._target.get(grp, obj.path, *args, **kwargs)
        else:
            return obj

    def put(self, grp, path,  *args, **kwargs):
        obj = self._mapper.get(path)

        if obj is None:
            self._target.put(grp, path, *args,  **kwargs)
        elif isinstance(obj,  Linker):
            self._target.put(grp, obj.path, *args, **kwargs)
        else:
            raise KeyError(f"Can not map path {path}")

    def iter(self, grp, path):
        for obj in self._mapper.iter(path):
            if isinstance(obj, Linker):
                yield self._target.get(grp, obj.path, **kwargs)
            else:
                yield obj
