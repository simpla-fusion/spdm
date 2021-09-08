import collections
import collections.abc
import functools
import typing
from functools import cached_property
from logging import log
from typing import Any, MutableSequence, Optional, Sequence

import numpy as np
from ..util.logger import logger
from .Entry import Entry, _not_found_
from .Node import Dict, List, Node, _next_,  _TObject, sp_property, _sp_property


def do_getattr(obj, k):
    if getattr(obj.__class__, '__getattr__', None) == do_getattr:
        raise RuntimeError(f"Recursive call")

    if k[0] == '_':
        bcls = obj.__class__.__bases__[0]
        res = bcls.__getattr__(obj, k)
    else:
        res = getattr(obj.__class__, k, _not_found_)

        if isinstance(res, property):
            res = getattr(res, "fget")(obj)
        elif isinstance(res, (_sp_property, cached_property)):
            res = res.__get__(obj)
        else:
            res = ht_get(obj, k, ignore_attribute=True)

    if isinstance(res, AttributeTree):
        pass
    elif res is _not_found_:
        res = _not_found_
    elif isinstance(res, Entry):
        res = AttributeTree(res)
    elif isinstance(res, (collections.abc.Mapping)):
        res = AttributeTree(res)

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
            if isinstance(obj, Node) and isinstance(obj._cache, collections.abc.MutableMapping):
                obj._cache[k] = v
                if hasattr(obj, k):
                    delattr(obj, k)
            else:
                raise AttributeError(f"Can not set cached_property '{k}'! ")
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


class AttributeTree(Dict[Node]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__new_child__ = AttributeTree

    def __getattr__(self, attr_name) -> Any:
        if attr_name[0] == '_':
            res = object.__getattr__(self, attr_name)
        else:
            res = self.__getitem__(attr_name)
        return res

    def __setattr__(self, attr_name: str, value: Any) -> None:
        if attr_name[0] == '_':
            object.__setattr__(self, attr_name, value)
        else:
            self.__setitem__(attr_name, value)

    def __delattr__(self, attr_name: str) -> None:
        return self.__delitem__(attr_name)


class AttributeCombine(List[_TObject]):
    def __init__(self,  *args, op=None,  initial_value=None, **kwargs) -> None:
        super().__init__(*args,  **kwargs)
        self._op = op
        self._initial_value = initial_value

    def __getattr__(self, attr_name) -> Any:
        return self.combine(attr_name, op=self._op, initial_value=self._initial_value)

    def __setattr__(self, attr_name: str, value: Any) -> None:
        raise NotImplementedError()

    def __delattr__(self, attr_name: str) -> None:
        raise NotImplementedError()
