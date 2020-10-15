
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
import pathlib
import collections
from xml.etree import (ElementTree, ElementInclude)
import numpy as np

Request = collections.namedtuple("Request", "path query fragment")


class Iterator(object):
    def __iter__(self):
        raise NotImplementedError()


class Holder(object):
    def __init__(self, data, *args, mode=None, **kargs):
        super().__init__()
        self._data = data

    @property
    def data(self):
        return self._data


class Handler(LazyProxy.Handler):
    DELIMITER = LazyProxy.DELIMITER

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def request(self, path, query={}, fragment=None):
        if isinstance(path, str):
            path = path.split(Handler.DELIMITER)
        # elif not isinstance(path, collections.abc.Sequence):
        #     raise TypeError(f"Illegal path type {type(path)}! {path}")
        return Request(path, query, fragment)


 


