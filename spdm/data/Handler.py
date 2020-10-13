
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
    def __init__(self, handler, wrapper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wrapper = wrapper
        self._handler = handler

    @property
    def handler(self):
        return self._handler

    def get(self, grp, path=[], *args, **kwargs):
        obj = self._wrapper.get(path)

        if obj is None:
            return self._handler.get(grp, path, *args, **kwargs)
        elif isinstance(obj, Linker):
            return self._handler.get(grp, obj.path, *args, **kwargs)
        else:
            return obj

    def put(self, grp, path,  *args, **kwargs):
        obj = self._wrapper.get(path)

        if obj is None:
            self._handler.put(grp, path, *args,  **kwargs)
        elif isinstance(obj,  Linker):
            self._handler.put(grp, obj.path, *args, **kwargs)
        else:
            raise KeyError(f"Can not map path {path}")

    def iter(self, grp, path):
        for obj in self._wrapper.iter(path):
            if not isinstance(obj, Linker):
                yield obj
            else:
                for item in self._handler.iter(grp, obj.path, **kwargs):
                    yield item
