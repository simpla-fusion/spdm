from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
import pathlib
import collections
from xml.etree import (ElementTree, ElementInclude)
from spdm.util.numlib import np
 

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

