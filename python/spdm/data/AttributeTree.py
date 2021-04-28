import collections
import functools
import typing

import numpy as np

from ..util.logger import logger
from .Node import Dict, List, Node, _TObject


def _getattr(self, k):
    if k.startswith("_"):
        return self.__dict__.get(k, None)
    else:
        res = getattr(self.__class__, k, None)
        if res is None:
            res = self.__getitem__(k)
        elif isinstance(res, property):
            res = getattr(res, "fget")(self)
        elif isinstance(res, functools.cached_property):
            res = res.__get__(self)

        return res


def _setattr(self, k, v):

    # pre_process = getattr(self.__class__, "__pre_process__", None)
    # if pre_process is not None:
    #     v = pre_process(self, v)

    if k.startswith("_"):
        self.__dict__[k] = v
    else:
        res = getattr(self.__class__, k, None)
        if res is None:
            self.__setitem__(k, v)
        elif isinstance(res, property):
            res.fset(self, k, v)
        elif isinstance(res, functools.cached_property):
            raise AttributeError(f"Can not set cached_property")
        elif isinstance(v, collections.abc.Mapping):
            target = _getattr(self, k)
            for i, d in v.items():
                _setattr(target, i, d)
        else:
            raise AttributeError(f"Can not set attribute {k}{type(d)}!")


def _delattr(self, k):
    if k.startswith("_"):
        del self.__dict__[k]
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


def as_attribute_tree(cls, *args, **kwargs):
    n_cls = type(f"{cls.__name__}__with_attr__", (cls,), {
        "__getattr__": _getattr,
        "__setattr__": _setattr,
        "__delattr__": _delattr,
    })

    return n_cls


@as_attribute_tree
class AttributeTree(Dict[str, _TObject]):
    # __slots__ = ()

    @classmethod
    def default_factory(cls, value, *args, **kwargs):
        if bool((Node.Category.DICT | Node.Category.ENTRY) & Node.type_category(value)):
            return AttributeTree(value, *args, **kwargs)
        else:
            return value

    def __init__(self, *args, default_factory=None, **kwargs):
        super().__init__(*args, default_factory=default_factory or AttributeTree.default_factory, **kwargs)

    # def __getattr__(self, k):
    #     if k in Node.__slots__:
    #         return getattr(self, k)
    #     elif k in self.__slots__:
    #         return super().__getattr__(k)
    #     else:
    #         res = getattr(self.__class__, k, None)
    #         if res is None:
    #             res = self.__getitem__(k)
    #         elif isinstance(res, property):
    #             res = getattr(res, "fget")(self)
    #         elif isinstance(res, functools.cached_property):
    #             res = res.__get__(self)

    #         return res

    # def __setattr__(self, k, v):
    #     if k in Node.__slots__:
    #         super(Node, self).__setattr__(k, v)
    #     elif k in self.__slots__:
    #         super().__setattr__(k, v)
    #     else:
    #         res = getattr(self.__class__, k, None)
    #         if res is None:
    #             self.__setitem__(k, v)
    #         elif isinstance(res, property):
    #             res.fset(self, k, v)
    #         elif isinstance(res, functools.cached_property):
    #             raise AttributeError(f"Can not set cached_property")
    #         elif isinstance(v, collections.abc.Mapping):
    #             target = self.getattr(self, k)
    #             for i, d in v.items():
    #                 target.__setattr__(target, i, d)
    #         else:
    #             raise AttributeError(f"Can not set attribute {k}:{type(v)}!")

    # def __delattr__(self, k):
    #     if k in Node.__slots__ or k in self.__slots__:
    #         raise AttributeError(k)
    #     else:
    #         res = getattr(self.__class__, k, None)
    #         if res is None:
    #             self.__delitem__(k)
    #         elif isinstance(res, property):
    #             res.fdel(self, k)
    #         elif isinstance(res, functools.cached_property):
    #             if k in self.__dict__:
    #                 del self.__dict__[k]
    #             # raise AttributeError(f"Can not set cached_property")
    #         else:
    #             raise AttributeError(f"Can not delete attribute {k}!")
