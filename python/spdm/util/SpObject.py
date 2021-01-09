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

_factory = None

_resolver = None


class SpObject(object):
    """
        Description:
            Super class of all spdm objects
        Attribute:
            parent      : parent node
            name        : short string
    """
    _default_prefix = ".".join(__package__.split('.')[:-1])

    _schema = "SpObject"
    _metadata = {"$class": "SpObject"}

    def __new__(cls, _metadata, *args,  **kwargs):
        if cls is not SpObject:
            return object.__new__(cls)

        if isinstance(_metadata, str):
            _metadata = {"$class": _metadata}
        elif not isinstance(_metadata, collections.abc.Mapping):
            raise TypeError(type(_metadata))

        n_cls = _metadata.get("$class", None)

        n_cls_name = None

        if isinstance(n_cls, str):
            if n_cls.startswith("."):
                n_cls = f"{SpObject._default_prefix}{n_cls}"
            n_cls_name = re.sub(r'[-\/\.\:]', '_', n_cls)
            n_cls = sp_find_module(n_cls)

        if inspect.isclass(n_cls):
            pass
        elif callable(n_cls):
            return n_cls(*args, _metadata=_metadata, **kwargs)
        else:
            raise ModuleNotFoundError(f"{_metadata}")

        n_cls = type(n_cls_name or f"{n_cls.__name__}_{uuid.uuid1()}", (n_cls,), {"_metadata": _metadata})

        obj = n_cls.__new__(_metadata, *args, **kwargs)
        obj._attributes = AttributeTree()
        return obj

    @classmethod
    def deserialize(cls, spec):
        if hasattr(cls, "_factory"):
            return cls._factory.create(spec)
        elif isinstance(spec, str):
            spec = cls._resolver.fetch(spec) if hasattr(cls, "_resolver") else io.read(spec)

        if not isinstance(spec, collections.abc.Mapping):
            raise TypeError(type(spec))
        spec.setdefault("name", spec.get("name", cls.__name__))
        return cls(**spec)

    def serialize(self):
        return {"$schema": self._schema,
                "$class": self._class,
                "$id":  self._oid,
                "attributes": self._attributes.__as_native__(),
                }

    def __init__(self,  *,   attributes=None, **kwargs):
        super().__init__()
        self._oid = uuid.uuid1()
        if isinstance(attributes, AttributeTree):
            self._attributes = attributes
        else:
            self._attributes = AttributeTree(collections.ChainMap(attributes or {}, kwargs))
        self._parent = None
        self._children = None

    # def __del__(self):
    #     if hasattr(self._parent, "remove_child"):
    #         self._parent.remove_child(self)

    @classmethod
    def from_json(cls, spec, *args, **kwargs):
        return cls.deserialize(spec, *args, **kwargs)

    def to_json(self):
        return self.serialize()

    @property
    def attributes(self):
        return self._attributes

    @cached_property
    def metadata(self):
        return AttributeTree(getattr(self.__class__, "_metadata", {}))

    def __hash__(self):
        return self._oid.int

    ##############################################################
    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return self._children

    def remove_child(self, child):
        raise NotImplementedError

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


def as_spobject(cls, *args, **kwargs):
    pass


__SP_EXPORT__ = SpObject
