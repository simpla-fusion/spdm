import collections
import functools
import inspect
import io
from logging import log
import uuid
from copy import deepcopy
from functools import cached_property
from typing import Type, TypeVar, Mapping
from ..data.AttributeTree import AttributeTree
from ..util.dict_util import deep_merge_dict
from ..util.logger import logger
from ..util.sp_export import sp_find_module, sp_find_module_by_name


_TSpObject = TypeVar('_TSpObject', bound='SpObject')


class SpObject(object):
    """
        Description:
            Super class of all spdm objects
        Attribute:
            parent      : parent node
            name        : short string
    """
    _default_prefix = ".".join(__package__.split('.')[:-1])

    association = {}

    def __init__(self, *args, **kwargs):
        super().__init__()
        if getattr(self, "_metadata", None) is None:
            self._metadata = {}

        # if len(kwargs) > 0:
        #     metadata = deep_merge_dict(metadata, kwargs)
        # else:
        #     metadata = deepcopy(metadata)

        # self._metadata: Mapping = metadata

    def serialize(self):
        return {"$schema": self._schema,
                "$class": self._class,
                "$id":  self._oid,
                }

    @classmethod
    def deserialize(cls, spec) -> _TSpObject:
        if hasattr(cls, "_factory"):
            return cls._factory.create(spec)
        elif isinstance(spec, str):
            spec = cls._resolver.fetch(spec) if hasattr(
                cls, "_resolver") else io.read(spec)

        if not isinstance(spec, collections.abc.Mapping):
            raise TypeError(type(spec))
        spec.setdefault("name", spec.get("name", cls.__name__))
        return cls(**spec)

    @classmethod
    def create(cls, metadata, *args,  **kwargs) -> _TSpObject:
        n_obj = SpObject.new_object(metadata)
        if isinstance(n_obj, SpObject):
            n_obj.__init__(*args, **kwargs)

        return n_obj

    @classmethod
    def new_object(cls, metadata) -> _TSpObject:
        if isinstance(metadata, str):
            metadata = {"$class": metadata}

        if not isinstance(metadata, collections.abc.Mapping):
            raise TypeError(type(metadata))

        n_cls = metadata.get("$class", None)

        if inspect.isclass(n_cls) or callable(n_cls):
            pass
        elif not isinstance(n_cls, str):
            raise RuntimeError(f"$class is not defined!")
        else:
            if n_cls.startswith("."):
                n_cls = SpObject.association.get(n_cls, n_cls)
            if n_cls.startswith("."):
                n_cls = f"spdm.plugins{n_cls}"
                metadata["$class"] = n_cls

            n_cls = sp_find_module_by_name(n_cls)

        if not inspect.isclass(n_cls):
            raise ModuleNotFoundError(metadata)

        obj = SpObject.__new__(n_cls)
        obj._metadata = metadata

        return obj

    @classmethod
    def from_json(cls, spec, *args, **kwargs):
        return cls.deserialize(spec, *args, **kwargs)

    def to_json(self):
        return self.serialize()

    @cached_property
    def metadata(self) -> AttributeTree:
        return AttributeTree(self._metadata)

    @cached_property
    def uuid(self) -> uuid. UUID:
        return uuid.uuid1()

    def __hash__(self):
        return self.uuid.int

    def __repr__(self):
        return f"<{self.__class__.__name__}   />"

    def __str__(self):
        return f"<{self.__class__.__name__}   />"


__SP_EXPORT__ = SpObject
