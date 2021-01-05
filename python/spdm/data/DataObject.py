import collections
import inspect
import pathlib
import pprint
import re
import os
import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.sp_export import sp_find_module
from spdm.util.SpObject import SpObject
from spdm.util.dict_util import format_string_recursive

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
            return SpObject.__new__(cls, data, *args, _metadata=_metadata, **kwargs)

        if isinstance(_metadata, str):
            n_cls = _metadata
        elif isinstance(_metadata, collections.abc.Mapping):
            n_cls = _metadata.get("$class", "general")
            if isinstance(n_cls, str):
                n_cls = n_cls.replace("/", ".").lower()
                if n_cls[0] != '.':
                    n_cls = DataObject.associations.get(n_cls, n_cls)
                    _metadata = collections.ChainMap({"$class": n_cls}, _metadata)
        else:
            n_cls = cls

        if n_cls in (int, float, str):
            return n_cls(data)
        else:
            return SpObject.__new__(cls, _metadata=_metadata)

    @classmethod
    def create(cls, data=None, *args, _metadata=None, envs=None, **kwargs):
        if _metadata is None and isinstance(data, collections.abc.Mapping) and ("$class" in data or "$schema" in data):
            _metadata = data
            data = None

        l_envs = collections.ChainMap(envs or {}, os.environ)

        if isinstance(data, str):
            data = format_string_recursive(data, l_envs)

        elif isinstance(data, collections.abc.Mapping):
            format_string_recursive(data,  l_envs)
            data = {k: DataObject.create(v, *args,  envs=envs, **kwargs) for k, v in data.items()}
        elif isinstance(data, collections.abc.Sequence):
            format_string_recursive(data,  l_envs)
            data = [DataObject.create(v, *args, envs=envs,  **kwargs) for v in data]

        if _metadata is None:
            return data

        if isinstance(_metadata, str):
            _metadata = {"$class": _metadata}
        elif not isinstance(_metadata, collections.abc.Mapping):
            raise TypeError(type(_metadata))

        n_cls = _metadata.get("$class", "")
        n_cls = n_cls.replace("/", ".").lower()
        n_cls = DataObject.associations.get(n_cls, n_cls)

        if isinstance(data, DataObject) and data.metadata["$class"] == n_cls:
            return data
        else:
            return DataObject(data, *args, _metadata=collections.ChainMap({"$class": n_cls}, _metadata), envs=envs,  **kwargs)

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
