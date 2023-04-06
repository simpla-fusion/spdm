import collections.abc
import enum
from functools import cached_property
from typing import Any, Generic, MutableSequence, Sequence

import numpy as np

from ..util.logger import logger
from ..util.misc import normalize_path, try_get
from .Entry import Entry
from .Function import Function
from .Node import Node


class Combiner(Entry):
    def __init__(self, cache: Sequence, *args,   **kwargs) -> None:
        super().__init__(cache, *args, **kwargs)

    def _get_data(self, path):
        path = self._prefix + normalize_path(path)
        if not path:
            return self._data
        else:
            cache = [try_get(d, path) for d in self._data]
            return [d for d in cache if d is not None]

    def get(self, path, *args, default_value=None, **kwargs):
        cache = self._get_data(path)
        if len(cache) == 0:
            return default_value
        elif all([isinstance(d, (np.ndarray, Function, float, int)) for d in cache]):
            return np.add.reduce([np.asarray(d) for d in cache])
        elif isinstance(cache[0], str):
            return cache[0]
        else:
            return Combiner(cache)

    def put(self, key, value: Any):
        raise NotImplementedError()

    def iter(self):
        cache = [try_get(d, self._prefix).__iter__() for d in self._data]
        if len(cache) == 0:
            return NotImplementedError()

        for d in cache[0]:
            yield Combiner([d, *[next(it) for it in cache[1:]]])


def combiner(*args, parent=None, **kwargs):
    return Node(Combiner(*args, **kwargs), parent=parent)
