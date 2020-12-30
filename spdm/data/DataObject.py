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

        schema["$id"] = schema_id

        metadata["$schema"] = schema

        return SpObject.__new__(cls, data, *args, metadata=metadata, **kwargs)

    # @staticmethod
    # def __new__(cls,  data=None, *args, schema=None, **kwargs):
    #     if cls is not DataObject:
    #         return object.__new__(cls)

    #     if isinstance(data, collections.abc.Mapping):
    #         schema = data.get("$schema", schema)

    #     if isinstance(schema, collections.abc.Mapping):
    #         schema_id = schema.get("$id", None)
    #         if data is None:
    #             data = schema.get("general_file", None)
    #     elif isinstance(schema, str):
    #         schema_id = schema
    #         schema = {"$id": schema_id}
    #     else:
    #         schema_id = None

    #     if schema_id == "integer":
    #         n_obj = int(data)
    #     elif schema_id == "float":
    #         n_obj = float(data)
    #     elif schema_id == "string":
    #         n_obj = str(data)
    #     elif schema_id == "ndarray":
    #         n_obj = load_ndarray(data, *args, schema=schema, **kwargs)
    #     elif schema is not None:
    #         n_cls = sp_find_module(f"{__package__}.{schema_id}")
    #         if n_cls is None:
    #             raise ModuleNotFoundError(f"{__package__}.{schema_id}")
    #         n_cls_name = re.sub(r'[-\/\.\:]', '_', schema_id)
    #         n_cls_name = f"{n_cls.__name__}_{n_cls_name}"
    #         n_cls = type(n_cls_name, (n_cls,), {"_metadata": {"schema": schema}})
    #         n_obj = SpObject.__new__(n_cls)
    #     elif isinstance(data, collections.abc.Mapping):
    #         n_obj = {k: DataObject(v) for k, v in data.items()}
    #     elif isinstance(data, list):
    #         n_obj = [DataObject(v) for v in data]
    #     else:
    #         logger.warning(f"Unknonw data type '{type(data)}'!")
    #         n_obj = data
    #     return n_obj

    def __init__(self, data, *args,  metadata=None,  **kwargs):
        super().__init__(*args, metadata=metadata,  **kwargs)
        # self.update(data)

    def serialize(self):
        return NotImplemented

    @classmethod
    def deserialize(cls, desc):
        return DataObject.__new__(cls, desc)

    # def __repr__(self):
    #     return pprint.pformat(self.metadata)

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
