import collections
import inspect
import pathlib
import pprint

import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.logger import logger
from spdm.util.sp_export import sp_find_module

from .Node import Node


def load_ndarray(desc, value, *args, **kwargs):
    if isinstance(value, np.ndarray):
        return value
    else:
        return NotImplemented


class DataObject(object):

    @staticmethod
    def __new__(cls,  desc, value=None, *args, **kwargs):
        if cls is not DataObject and cls.__name__ not in ('Document', 'Collection'):
            return super(DataObject, cls).__new__(cls)

        if isinstance(desc, str):
            desc = {"schema": desc}
        elif isinstance(desc, pathlib.PosixPath):
            desc = {"schema": "local", "path": desc}
        elif isinstance(desc, collections.abc.Sequence):
            desc = {"schema": ".".join(desc)}

        if not isinstance(desc, (AttributeTree)):
            desc = AttributeTree(desc)
            # raise TypeError(f"Illegal type! 'desc' {type(desc)}")

        d_schema = desc.schema or "string"

        desc["schema"] = d_schema

        if value is None:
            value = desc.default or None

        if d_schema == "integer":
            n_obj = int(value)
        elif d_schema == "float":
            n_obj = float(value)
        elif d_schema == "string":
            n_obj = str(value)
        elif d_schema == "ndarray":
            n_obj = load_ndarray(desc, value, *args, **kwargs)
        else:
            mod_path = f"{__package__}.{d_schema.replace('/','.')}"

            n_cls = sp_find_module(mod_path)

            if inspect.isclass(n_cls):
                n_obj = object.__new__(n_cls)
            elif callable(n_cls):
                n_obj = n_cls(desc, value, *args, **kwargs)
            else:
                raise ValueError(f"Illegal DataObject maker! {type(n_cls)}")

        return n_obj

    def __init__(self, desc, value=None, *args, **kwargs):
        self._description = AttributeTree(desc)
        if value is not None:
            self.update(value)

    def serialize(self):
        return NotImplemented

    @classmethod
    def deserialize(cls, desc):
        return DataObject.__new__(cls, desc)

    def __repr__(self):
        return pprint.pformat(self.description)

    @property
    def description(self):
        return self._description

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
