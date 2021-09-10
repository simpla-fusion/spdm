import collections
import inspect
import io
import uuid
from copy import deepcopy
from functools import cached_property
from typing import TypeVar, Mapping
from ..data.AttributeTree import AttributeTree
from ..util.dict_util import deep_merge_dict
from ..util.logger import logger
from ..util.sp_export import sp_find_module


class SpObject(object):
    """
        Description:
            Super class of all spdm objects
        Attribute:
            parent      : parent node
            name        : short string
    """
    _default_prefix = ".".join(__package__.split('.')[:-1])

    def __init__(self, metadata=None, /, **kwargs):
        super().__init__()

        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, collections.abc.Mapping):
            raise TypeError(f"Illegal metadata {type(metadata)}")

        if len(kwargs) > 0:
            metadata = deep_merge_dict(metadata, kwargs)
        else:
            metadata = deepcopy(metadata)

        self._metadata: Mapping = metadata

    @classmethod
    def deserialize(cls, spec):
        # @classmethod
        # def parse_metadata(cls,  metadata=None, *args, **kwargs):

        # return n_cls

        if hasattr(cls, "_factory"):
            return cls._factory.create(spec)
        elif isinstance(spec, str):
            spec = cls._resolver.fetch(spec) if hasattr(
                cls, "_resolver") else io.read(spec)

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


def create_object(metadata, *args, **kwargs) -> object:
    if isinstance(metadata, str):
        cls_name = metadata
    elif isinstance(metadata, collections.abc.Mapping):
        cls_name = metadata.get("$class")

    n_cls = None
    if inspect.isclass(cls_name):
        n_cls = cls_name
    elif isinstance(cls_name, str):
        if cls_name.startswith("."):
            cls_name = f"{SpObject._default_prefix}{cls_name}"
        n_cls = sp_find_module(cls_name)

    if inspect.isclass(cls_name) or callable(n_cls):
        pass
    else:
        raise ModuleNotFoundError(metadata)

    return n_cls(*args, **kwargs)


__SP_EXPORT__ = SpObject
