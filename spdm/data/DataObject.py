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
        "file.general": ".data.file.GeneralFile",
        "file.bin": ".data.file.Binary",
        "file.h5": ".data.file.HDF5",
        "file.hdf5": ".data.file.HDF5",
        "file.nc": ".data.file.netCDF",
        "file.netcdf": ".data.file.netCDF",
        "file.namelist": ".data.file.namelist",
        "file.nml": ".data.file.namelist",
        "file.xml": ".data.file.XML",
        "file.json": ".data.file.JSON",
        "file.yaml": ".data.file.YAML",
        "file.txt": ".data.file.TXT",
        "file.csv": ".data.file.CSV",
        "file.numpy": ".data.file.NumPy",
        "file.geqdsk": ".data.file.GEQdsk",
        "file.gfile": ".data.file.GEQdsk",
        "file.mdsplus": ".data.db.MDSplus#MDSplusDocument",

        "integer": int,
        "float": float,
        "string": str,
        "ndarray": np.ndarray
    }

    @staticmethod
    def __new__(cls, data=None,  *args, metadata=None, **kwargs):
        if cls is not DataObject and cls.__name__ not in ("File"):
            return object.__new__(cls)

        if metadata is None and isinstance(data, collections.abc.Mapping) and "$schema" in data:
            metadata = data
            data = None
        elif isinstance(metadata, str):
            metadata = {"$schema": metadata}
        elif metadata is None:
            metadata = {}
        elif not isinstance(metadata, collections.abc.Mapping):
            raise TypeError(type(metadata))

        schema = metadata.get("$schema", {})

        if isinstance(schema, str):
            schema = {"$id": schema}

        if not isinstance(schema, collections.abc.Mapping):
            raise TypeError(type(schema))

        schema_id = schema.get("$id",  "general")

        schema_id = schema_id.replace('/', '.').strip(".")

        if not schema_id.startswith("."):
            schema_id = DataObject.associations.get(schema_id.lower(), schema_id)

        if schema_id in (int, float, str):
            if data is None:
                data = metadata.get("default", None) or schema.get("default", 0)
            return schema_id(data)
        elif inspect.isclass(schema_id):
            return object.__new__(schema_id)
        else:
            schema["$id"] = schema_id

            metadata["$schema"] = schema

            return SpObject.__new__(cls, data, *args, metadata=metadata, **kwargs)

    def __init__(self, data=None, *args,  metadata=None,  **kwargs):
        super().__init__(*args, metadata=metadata,  **kwargs)

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def serialize(self):
        return NotImplemented

    @classmethod
    def deserialize(cls, desc):
        return DataObject.__new__(cls, desc)

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
