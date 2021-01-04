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
        "ndarray": np.ndarray,

        "file.general": ".data.file.GeneralFile",
        "file.bin": ".data.file.Binary",
        "file.hdf5": ".data.file.HDF5",
        "file.netcdf": ".data.file.netCDF",
        "file.namelist": ".data.file.namelist",
        "file.xml": ".data.file.XML",
        "file.json": ".data.file.JSON",
        "file.yaml": ".data.file.YAML",
        "file.txt": ".data.file.TXT",
        "file.csv": ".data.file.CSV",
        "file.numpy": ".data.file.NumPy",
        "file.geqdsk": ".data.file.GEQdsk",
        "file.mdsplus": ".data.db.MDSplus#MDSplusDocument",
    }

    @staticmethod
    def __new__(cls, data=None, *args, _metadata=None, **kwargs):
        if cls is not DataObject and _metadata is None:
            # return object.__new__(cls)
            return SpObject.__new__(cls, data, *args, _metadata=_metadata, **kwargs)

        if isinstance(_metadata, collections.abc.Mapping):
            n_cls = _metadata.get("$class", None)
            n_cls = n_cls.replace("/", ".").lower()
            n_cls = DataObject.associations.get(n_cls, n_cls)
            _metadata = collections.ChainMap({"$class": n_cls}, _metadata)
        else:
            n_cls = cls

        if n_cls in (int, float, str):
            return n_cls(data)
        else:
            # if not not _metadata:  # isinstance(_metadata, (collections.abc.Mapping, str)):
            return SpObject.__new__(cls, _metadata=_metadata)

        # if isinstance(_metadata, str):
        #     _metadata = {"$schema": _metadata}
        # elif _metadata is None:
        #     _metadata = {}
        # elif not isinstance(_metadata, collections.abc.Mapping):
        #     raise TypeError(type(_metadata))

        # schema = _metadata.get("$schema", {})

        # if isinstance(schema, str):
        #     schema = {"$id": schema}

        # if not isinstance(schema, collections.abc.Mapping):
        #     raise TypeError(type(schema))

        # schema_id = schema.get("$id",  "general")

        # schema_id = schema_id.replace('/', '.').strip(".")

        # n_cls = _metadata.get("$class", None) or DataObject.associations.get(schema_id.lower(), None)

        # if n_cls in (int, float, str):
        #     if data is None:
        #         data = _metadata.get("default", None) or schema.get("default", 0)
        #     return n_cls(data)
        # elif inspect.isclass(n_cls):
        #     return object.__new__(n_cls)
        # else:
        #     _metadata["$class"] = n_cls

        #     return SpObject.__new__(cls, data, *args, _metadata=_metadata, **kwargs)

    @classmethod
    def create(cls, data=None, *args, _metadata=None, **kwargs):
        if _metadata is None and isinstance(data, collections.abc.Mapping) and ("$class" in data or "$schema" in data):
            _metadata = data
            data = None

        if _metadata is not None:
            return DataObject(data, *args, _metadata=_metadata, **kwargs)
        elif isinstance(data, collections.abc.Mapping):
            return {k: DataObject.create(v, *args, **kwargs) for k, v in data.items()}
        elif isinstance(data, collections.abc.Sequence) and not isinstance(data, str):
            return [DataObject.create(v, *args, **kwargs) for v in data]
        else:
            return data

    def __init__(self, data=None, *args,  _metadata=None,  **kwargs):
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
