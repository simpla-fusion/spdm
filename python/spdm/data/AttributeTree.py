import collections
import functools
import typing

import numpy as np

from ..util.logger import logger
from .Node import Dict, List, Node, _TObject
from .Node import _next_


class AttributeTree(Dict[str, _TObject]):
    @classmethod
    def default_factory(cls, value, *args, **kwargs):
        if bool((Node.Category.DICT | Node.Category.ENTRY) & Node.__type_category__(value)):
            return AttributeTree(value, *args, **kwargs)
        else:
            return value

    def __init__(self, *args, default_factory=None, **kwargs):
        super().__init__(*args, default_factory=default_factory or AttributeTree.default_factory, **kwargs)

    def __getattr__(self, k):
        if k[0] == '_':
            return super().__getattr__(self, k)
        elif k in Node.__slots__:
            res = getattr(self, k)
        elif k in self.__slots__:
            res = super().__getattr__(k)
        elif k[0] == '_':
            # if hasattr(self, '__dict__') and k in self.__dict__:
            try:
                return self.__dict__[k]
            except Exception:
                raise AttributeError(k)
        else:
            res = getattr(self.__class__, k, None)
            if hasattr(self.__class__, 'get'):
                res = self.get(k, None)
            elif res is None:
                res = self.__getitem__(k)
            elif isinstance(res, property):
                res = getattr(res, "fget")(self)
            elif isinstance(res, functools.cached_property):
                res = res.__get__(self)
        if res is None:
            return AttributeTree(Node.LazyHolder(self, [k]))
        elif isinstance(res, (collections.abc.Mapping, Node)):
            return AttributeTree(res)
        else:
            return res

    def __setattr__(self, k, v):
        if k[0] == '_':
            super().__setattr__( k, v)
        elif k in Node.__slots__:
            Node.__setattr__(self, k, v)
        elif k in self.__slots__:
            super().__setattr__(k, v)
        else:
            res = getattr(self.__class__, k, None)
            if res is None:
                self.__setitem__(k, v)
            elif isinstance(res, property):
                res.fset(self, k, v)
            elif isinstance(res, functools.cached_property):
                raise AttributeError(f"Can not set cached_property")
            elif isinstance(v, collections.abc.Mapping):
                target = self.__getattr__(k)
                for i, d in v.items():
                    target.__setattr__(target, i, d)
            else:
                raise AttributeError(f"Can not set attribute {k}:{type(v)}!")

    def __delattr__(self, k):
        if k in Node.__slots__ or k in self.__slots__:
            raise AttributeError(k)
        else:
            res = getattr(self.__class__, k, None)
            if res is None:
                self.__delitem__(k)
            elif isinstance(res, property):
                res.fdel(self, k)
            elif isinstance(res, functools.cached_property):
                if k in self.__dict__:
                    del self.__dict__[k]
                # raise AttributeError(f"Can not set cached_property")
            else:
                raise AttributeError(f"Can not delete attribute {k}!")

    def __iter__(self) -> typing.Iterator[Node]:
        # for v in super(Node, self).__iter__():
        #     yield AttributeTree({v})
        yield from Node.__iter__(self)


def as_attribute_tree(cls, *args, **kwargs):
    n_cls = type(f"{cls.__name__}__with_attr__", (cls,), {
        "__getattr__": AttributeTree.__getattr__,
        "__setattr__": AttributeTree.__setattr__,
        "__delattr__": AttributeTree.__delattr__,
        "__iter__": AttributeTree.__iter__,
    })
    return n_cls
