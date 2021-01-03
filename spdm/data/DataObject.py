import collections
import inspect
import pathlib
import pprint
import re
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

    associations = {
        "general": ".data.General",
        "integer": int,
        "float": float,
        "string": str,
        "ndarray": np.ndarray
    }

    @staticmethod
    def __new__(cls, data=None,  *args, metadata=None, **kwargs):
        if cls is not DataObject and metadata is None:
            return SpObject.__new__(cls, data,  *args, **kwargs)

        if isinstance(metadata, collections.abc.Mapping):
            return super(cls, SpObject).deserialize(metadata)
        elif isinstance(metadata, str):
            raise NotImplementedError()
        elif isinstance(data, str):
            return data
        elif isinstance(data, collections.abc.Mapping):
            return {k: DataObject(v) for k, v in data.items()}
        elif isinstance(data, collections.abc.Sequence):
            return [DataObject(v) for v in data]
        else:
            return data

        # if isinstance(metadata, str):
        #     metadata = {"$schema": metadata}
        # elif metadata is None:
        #     metadata = {}
        # elif not isinstance(metadata, collections.abc.Mapping):
        #     raise TypeError(type(metadata))

        # schema = metadata.get("$schema", {})

        # if isinstance(schema, str):
        #     schema = {"$id": schema}

        # if not isinstance(schema, collections.abc.Mapping):
        #     raise TypeError(type(schema))

        # schema_id = schema.get("$id",  "general")

        # schema_id = schema_id.replace('/', '.').strip(".")

        # n_cls = metadata.get("$class", None) or DataObject.associations.get(schema_id.lower(), None)

        # if n_cls in (int, float, str):
        #     if data is None:
        #         data = metadata.get("default", None) or schema.get("default", 0)
        #     return n_cls(data)
        # elif inspect.isclass(n_cls):
        #     return object.__new__(n_cls)
        # else:
        #     metadata["$class"] = n_cls

        #     return SpObject.__new__(cls, data, *args, metadata=metadata, **kwargs)

    def __init__(self, data=None, *args,  metadata=None,  **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def serialize(self, *args, **kwargs):
        return super().serialize(*args, **kwargs)

    @classmethod
    def deserialize(cls, *args, **kwargs):
        return super().deserialize(cls, *args, **kwargs)

    @property
    def root(self):
        return Node(self)

    @property
    def entry(self):
        return self.root.entry

    @property
    def value(self):
        return NotImplemented

    def update(self, value):
        raise NotImplementedError
