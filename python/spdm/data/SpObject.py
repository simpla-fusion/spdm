from __future__ import annotations

import collections
import collections.abc
import inspect

from ..common.tags import _not_found_
from ..util.sp_export import sp_load_module

SP_MODULE_NAME = "spdm"


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

    @classmethod
    def create(cls, n_cls: str) -> SpObject:
        if not isinstance(n_cls, str):
            raise TypeError(f"$class is not a string! {n_cls}")

        if n_cls.startswith('.'):
            n_cls = f"{cls.__module__.lower()}{n_cls}"

        n_module_name = SpObject.association.get(n_cls, _not_found_)

        if inspect.isclass(n_module_name):
            n_module = n_module_name
        elif isinstance(n_module_name, str):
            if n_module_name.startswith("."):
                n_module_name = f"{SP_MODULE_NAME}.plugins{n_module_name}"
            n_module = sp_load_module(n_module_name)
        else:
            raise ModuleNotFoundError(n_cls)

        if inspect.isclass(n_module):
            return object.__new__(n_module)
        elif callable(n_module):
            return n_module()
        else:
            raise ModuleNotFoundError(n_cls)

    def serialize(self) -> dict:
        return {"$class": str(self.__class__)}

    @classmethod
    def deserialize(cls, spec: collections.abc.Mapping) -> _TSpObject:
        if not isinstance(spec, collections.abc.Mapping) or "$class" not in spec:
            raise ValueError(spec)

        return SpObject.create(spec.get("$class"))

    def to_json(self) -> dict:
        return self.serialize()

    @classmethod
    def from_json(cls, spec: collections.abc.Mapping) -> _TSpObject:
        return cls.deserialize(spec)

    # @property
    # def metadata(self):
    #     return self._metadata

    # @cached_property
    # def uuid(self) -> uuid. UUID:
    #     return uuid.uuid1()

    # def __hash__(self):
    #     return self.uuid.int

    def __repr__(self):
        return f"<{self.__class__.__name__}   />"

    def __str__(self):
        return f"<{self.__class__.__name__}   />"


__SP_EXPORT__ = SpObject
