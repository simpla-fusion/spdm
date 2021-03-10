import collections
import inspect
import os
import pathlib
import pprint
from typing import Type

import numpy as np
from spdm.util.logger import logger
from spdm.util.SpObject import SpObject
from spdm.util.urilib import urisplit

from .Entry import Entry


def load_ndarray(desc, value, *args, **kwargs):
    if isinstance(value, np.ndarray):
        return value
    else:
        return NotImplemented


class DataObject(SpObject):

    schema = {
        "general": ".data.General",
        "integer": int,
        "float": float,
        "string": str,
        "ndarray": np.ndarray,
    }

    @staticmethod
    def __new__(cls,  metadata=None, *args, **kwargs):
        if isinstance(metadata, collections.abc.Mapping):
            n_cls = metadata.get("$class", None)
        elif isinstance(metadata, str):
            n_cls = urisplit(metadata)["schema"] or metadata
        else:
            n_cls = None

        if cls is not DataObject and not n_cls:
            return SpObject.__new__(cls)

        if inspect.isclass(n_cls):
            return object.__new__(n_cls)

        if isinstance(n_cls, str) and not n_cls.startswith("."):
            n_cls = DataObject.schema.get(n_cls.lower(), None)

        return SpObject.__new__(cls, n_cls)

    def __init__(self, metadata=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metadata = metadata

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def serialize(self, *args, **kwargs):
        return super().serialize(*args, **kwargs)

    @classmethod
    def deserialize(cls, *args, **kwargs):
        return super().deserialize(cls, *args, **kwargs)

    @property
    def metadata(self):
        return self._metadata

    @property
    def entry(self):
        return Entry(self)

    @property
    def value(self):
        return NotImplemented

    def update(self, value):
        raise NotImplementedError
