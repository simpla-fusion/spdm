import collections
import copy
import inspect
import re
import traceback
import uuid
from enum import Enum, Flag, auto, unique

from .AttributeTree import AttributeTree
from .logger import logger
from .sp_export import sp_find_module
from .urilib import urisplit


class SpObject(object):
    """
        Description:
            Super class of all spdm objects
        Attribute:
            parent      : parent node
            name        : short string
    """

    @classmethod
    def new_class(cls,  desc=None, *args, ** kwargs):

        description = AttributeTree(getattr(cls, "_description", {}))

        if desc is None and len(kwargs) == 0:
            return cls
        elif isinstance(desc, str):
            desc = {"$id": desc}

        description.__update__(desc)
        description.__update__(kwargs)

        # if factory is not None:
        #     description = factory.resolver.fetch(description)

        # if not base_class:
        #     base_class = cls
        # elif factory is not None:
        #     base_class = factory.new_class(base_class)
        # else:
        #     base_class = cls
        # base_class = description["$base_class"]

        o = urisplit(description["$id"] or f"{cls.__name__}_{uuid.uuid1().hex}")
        n_cls_name = f"{o.authority.replace('.', '_')}_" if o.authority is not None else ""

        path = re.sub(r'[-\/\.]', '_', o.path)
        
        n_cls_name = f"{n_cls_name}{path}"
        
        n_cls = type(n_cls_name, (cls,), {"_description": AttributeTree(description)})

        return n_cls

    @classmethod
    def create(cls, *args,  ** kwargs):
        # key starts with '_' are resevered for classmethod new_class
        c_kwargs = {}
        o_kwargs = {}
        for k, v in kwargs.items():
            if k.startswith("_"):
                c_kwargs[k[1:]] = v
            else:
                o_kwargs[k] = v

        return cls._create_from((args, o_kwargs), **c_kwargs)

    @classmethod
    def _create_from(cls, init_args=None, *_args, **_kwargs):
        if len(_args) > 0 or len(_kwargs) > 0:
            cls = cls.new_class(*_args, **_kwargs)

        args = []
        kwargs = {}
        if init_args is None:
            pass
        elif isinstance(init_args, collections.abc.Mapping):
            kwargs = init_args
        elif isinstance(init_args, list):
            args = init_args
        elif isinstance(init_args, tuple):
            args, kwargs = init_args
        else:
            args = [init_args]

        return cls(*args, **kwargs)

    def __init__(self,  *, name=None, uid=None, label=None, parent=None, attributes=None, **kwargs):
        super().__init__()
        self._uuid = uid or uuid.uuid1()
        self._parent = parent
        self._attributes = AttributeTree(collections.ChainMap(attributes or {}, kwargs))
        self._name = name
        self._label = label
        logger.debug(f"Initialize: {self.__class__.__name__} ")

    def __del__(self):
        logger.debug(f"Finialize: {self.__class__.__name__} ")

    # def __del__(self):
    #     # p = getattr(self, "_parent", None)
    #     # if p is not None:
    #     #     p.remove_child(self)
    #     pass

    def preprocess(self):
        logger.debug(f"Preprocess: {self.__class__.__name__}")

    def postprocess(self):
        logger.debug(f"Postprocess: {self.__class__.__name__}")

    def execute(self, *args, **kwargs):
        logger.debug(f"Execute: {self.__class__.__name__}")
        return None

    def __call__(self, *args, **kwargs):
        self.preprocess()

        error_msg = None
        # try:
        res = self.execute(*args, **kwargs)
        # except Exception as error:
        #     error_msg = error
        #     logger.error(f"{error}")
        #     res = None

        self.postprocess()

        # if error_msg is not None:
        #     raise error_msg

        return res

    def __hash__(self):
        return self._uuid.int

    ##############################################################
    @property
    def parent(self):
        return self._parent

    @property
    def is_root(self):
        return self._parent is None

    @property
    def rank(self):
        return self._parent.rank+1 if self._parent is not None else 0

    @property
    def attributes(self):
        return self._attributes

    @property
    def name(self):
        return self._name

    @property
    def kind(self):
        return self.__class__.__name__

    @property
    def full_name(self):
        if self._parent is not None:
            return f"{self._parent.full_name}.{self._name}"
        else:
            return self._name

    def shorten_name(self, num=2):
        s = self.full_name.split('.')
        if len(s) <= num:
            return '.'.join(s)
        else:
            return f"{s[0]}...{'.'.join(s[-(num-1):])}"

    @property
    def uuid(self):
        return self._uuid

    @property
    def id(self):
        return self._uuid.int

    @property
    def label(self):
        return self._label

    def __repr__(self):
        return f"<{self.__class__.__name__} path=\"{self.full_name}\"   />"

    @classmethod
    def deserialize(cls, spec: collections.abc.Mapping):
        spec = spec or {}
        if not isinstance(spec, collections.abc.Mapping):
            raise TypeError(type(spec).__name__)
        spec.setdefault("name", spec.get("name", cls.__name__))
        return cls(**spec)

    def serialize(self):
        return {"$schema": "SpObject",
                "$class": f"{self.__class__.__module__}.{self.__class__.__name__}",
                "annotation": {"label": self.label},
                "metadata": {}}

    @classmethod
    def from_json(cls, spec, *args, **kwargs):
        return cls.deserialize(spec, *args, **kwargs)

    def to_json(self):
        return self.serialize()

    ######################################################
    # algorithm
    def find_shortest_path(self, s, t):
        return []


__SP_EXPORT__ = SpObject
