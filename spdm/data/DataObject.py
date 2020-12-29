import collections
import inspect
import pathlib
import pprint

import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.sp_export import sp_find_module
from spdm.util.SpObject import SpObject

from .Node import Node


def load_ndarray(desc, value, *args, **kwargs):
    if isinstance(value, np.ndarray):
        return value
    else:
        return NotImplemented


class DataObject(SpObject):

    @staticmethod
    def __new__(cls,  metadata, value=None, *args, **kwargs):
        if cls is not DataObject:
            return super().__new__(metadata, value=None, *args, **kwargs)

        if isinstance(metadata, str):
            metadata = {"$schema": metadata}
        elif isinstance(metadata, pathlib.PosixPath):
            metadata = {"$schema": "local", "path": metadata}
        elif isinstance(metadata, collections.abc.Sequence):
            metadata = {"$schema": ".".join(metadata)}

        d_schema = metadata.get("$schema", "string").replace("/",".")

        metadata["$schema"] = d_schema

        if value is None:
            value = metadata.get("default", None)

        if d_schema == "integer":
            n_obj = int(value)
        elif d_schema == "float":
            n_obj = float(value)
        elif d_schema == "string":
            n_obj = str(value)
        elif d_schema == "ndarray":
            n_obj = load_ndarray(metadata, value, *args, **kwargs)
        else:
            n_cls = sp_find_module(f"{__package__}.{d_schema}")
            if n_cls is None:
                raise ModuleNotFoundError(f"{__package__}.{d_schema}")
            n_obj = SpObject.__new__(n_cls)
        return n_obj

    def __init__(self, metadata, value=None, *args, **kwargs):
        if isinstance(metadata, str):
            metadata = {"$schema": metadata}
        super().__init__(*args, attributes=metadata, **kwargs)

        if value is not None:
            self.update(value)

    def serialize(self):
        return NotImplemented

    @ classmethod
    def deserialize(cls, desc):
        return DataObject.__new__(cls, desc)

    def __repr__(self):
        return pprint.pformat(self.metadata)

    @ property
    def root(self):
        return Node(self)

    @ property
    def entry(self):
        return self.root.entry

    @ property
    def value(self):
        return NotImplemented

    def update(self, value):
        raise NotImplementedError
