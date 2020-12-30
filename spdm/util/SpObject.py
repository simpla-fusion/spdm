import collections
import copy
import inspect
import re
import traceback
import uuid
from functools import cached_property

import numpy as np

from .AttributeTree import AttributeTree
from .logger import logger
from .sp_export import sp_find_module


class SpObject(object):
    """
        Description:
            Super class of all spdm objects
        Attribute:
            parent      : parent node
            name        : short string
    """
    _default_prefix = ".".join(__package__.split('.')[:-1])

    _metadata = {"$schema": "SpObject"}

    @staticmethod
    def __new__(cls, data=None, *args, metadata=None, **kwargs):
        # if cls is not SpObject:
        #     return object.__new__(cls)

        if isinstance(metadata, str):
            metadata = {"$schema": metadata}
        elif not isinstance(metadata, collections.abc.Mapping):
            raise TypeError(type(metadata))

        schema = metadata.get("$schema", None)

        if isinstance(schema, str):
            schema_id = schema
            schema = {"$id": schema_id}
        elif isinstance(schema, collections.abc.Mapping):
            schema_id = schema.get("$id", None)
        else:
            raise TypeError(type(schema))

        if schema_id.startswith("."):
            schema_id = f"{SpObject._default_prefix}{schema_id}"

        n_cls = sp_find_module(schema_id)

        if n_cls is None:
            raise ModuleNotFoundError(f"{schema_id}")
        n_cls_name = re.sub(r'[-\/\.\:]', '_', schema_id)
        # n_cls_name = f"{n_cls.__name__}_{n_cls_name}"
        n_cls = type(n_cls_name, (n_cls,), {"_metadata": metadata})

        if inspect.isclass(n_cls):
            return object.__new__(n_cls)
        elif callable(n_cls):
            return n_cls(data, *args, metadata=metadata, **kwargs)
        else:
            raise RuntimeError(f"Illegal SpObject type! {type(n_cls)}")

    def __init__(self,  *,  oid=None,  parent=None, attributes=None, metadata=None, **kwargs):
        super().__init__()
        self._oid = oid or uuid.uuid1()
        self._parent = parent
        self._attributes = AttributeTree(collections.ChainMap(attributes or {}, kwargs))

    # def __del__(self):
    #     if hasattr(self._parent, "remove_child"):
    #         self._parent.remove_child(self)

    @classmethod
    def deserialize(cls, spec: collections.abc.Mapping):
        spec = spec or {}
        if not isinstance(spec, collections.abc.Mapping):
            raise TypeError(type(spec).__name__)
        spec.setdefault("name", spec.get("name", cls.__name__))
        return cls(**spec)

    def serialize(self):
        return {"$schema": self.metadata["$schema"],
                "$id":  self.metadata["$id"] or f"{self.__class__.__module__}.{self.__class__.__name__}",
                "attributes": self._attributes,
                "metadata": {}}

    @classmethod
    def from_json(cls, spec, *args, **kwargs):
        return cls.deserialize(spec, *args, **kwargs)

    def to_json(self):
        return self.serialize()

    @cached_property
    def metadata(self):
        return AttributeTree(self.__class__._metadata)

    @property
    def attributes(self):
        return self._attributes

    def __hash__(self):
        return self._oid.int

    ##############################################################
    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return []

    def remove_child(self, child):
        pass

    def insert_child(self, child, name=None, *args, **kwargs):
        raise NotImplementedError

    @property
    def is_root(self):
        return self._parent is None

    @property
    def rank(self):
        return self._parent.rank+1 if self._parent is not None else 0

    @property
    def kind(self):
        return self.__class__.__name__

    @property
    def full_name(self):
        if self._parent is not None:
            return f"{self._parent.full_name}.{str(self.attributes.name)}"
        else:
            return str(self.attributes.name)

    def shorten_name(self, num=2):
        s = self.full_name.split('.')
        if len(s) <= num:
            return '.'.join(s)
        else:
            return f"{s[0]}...{'.'.join(s[-(num-1):])}"

    @property
    def oid(self):
        return self._oid

    def __repr__(self):
        return f"<{self.__class__.__name__}   />"

    def __str__(self):
        return f"<{self.__class__.__name__}   />"

    ######################################################
    # algorithm

    def find_shortest_path(self, s, t):
        return []


__SP_EXPORT__ = SpObject
