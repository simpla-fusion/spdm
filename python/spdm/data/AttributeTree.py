import collections
import functools
from logging import log
import typing

import numpy as np

from ..util.logger import logger
from .Node import Dict, List, Node, _TObject
from .Node import _next_


def do_getattr(obj, k):
    if k[0] == '_':
        scls = obj.__class__
        bcls = obj.__class__.__bases__[0]
        obj.__class__ = bcls
        res = getattr(obj, k)
        obj.__class__ = scls
    else:
        res = getattr(obj.__class__, k, None)

        if isinstance(res, property):
            res = getattr(res, "fget")(obj)
        elif isinstance(res, functools.cached_property):
            res = res.__get__(obj)
        elif hasattr(obj.__class__, 'get'):
            res = obj.get(k, None)
        else:
            res = obj.__getitem__(k)
    if res is None:
        return AttributeTree(Node.LazyHolder(obj, [k]))
    elif isinstance(res, (collections.abc.Mapping, Node)):
        return AttributeTree(res)
    else:
        return res


def do_setattr(obj, k, v):
    if k[0] == '_':
        object.__setattr__(obj, k, v)
    else:
        res = getattr(obj.__class__, k, None)
        if res is None:
            obj.__setitem__(k, v)
        elif isinstance(res, property):
            res.fset(obj, k, v)
        elif isinstance(res, functools.cached_property):
            raise AttributeError(f"Can not set cached_property")
        elif isinstance(v, collections.abc.Mapping):
            target = obj.__getattr__(k)
            for i, d in v.items():
                target.__setattr__(target, i, d)
        else:
            raise AttributeError(f"Can not set attribute {k}:{type(v)}!")


def do_delattr(obj, k):
    if k in Node.__slots__ or k in obj.__slots__:
        raise AttributeError(k)
    else:
        res = getattr(obj.__class__, k, None)
        if res is None:
            obj.__delitem__(k)
        elif isinstance(res, property):
            res.fdel(obj, k)
        elif isinstance(res, functools.cached_property):
            if k in obj.__dict__:
                del obj.__dict__[k]
            # raise AttributeError(f"Can not set cached_property")
        else:
            raise AttributeError(f"Can not delete attribute {k}!")


def as_attribute_tree(cls, *args, **kwargs):
    n_cls = type(f"{cls.__name__}__with_attr__", (cls,), {
        "__getattr__": do_getattr,
        "__setattr__": do_setattr,
        "__delattr__": do_delattr,
        # "__iter__": do_iter,
    })
    return n_cls


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
        return do_getattr(self, k)

    def __setattr__(self, k, v):
        do_setattr(self, k, v)

    def __delattr__(self, k):
        do_delattr(self, k)

    def __iter__(self) -> typing.Iterator[Node]:
        # for v in super(Node, self).__iter__():
        #     yield AttributeTree({v})
        logger.debug(self.__class__)
        yield from Node.__iter__(self)
