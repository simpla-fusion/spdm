import collections.abc
from _thread import RLock
from typing import Generic

import numpy as np

from ..util.utilities import _not_found_
from .Function import Function
from .Node import Node, _TObject


class sp_cached_property(Generic[_TObject]):
    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.lock = RLock()

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def __get__(self, instance: Node, owner=None) -> _TObject:
        if not isinstance(instance, Node):
            raise NotImplemented

        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it.")

        try:
            cache = instance._entry
        except AttributeError:  # not all objects have __dict__ (e.g. class defines slots)
            msg = (
                f"No '_entry' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None

        val = cache.get(self.attrname, _not_found_)
        if not isinstance(val, (int, str, np.ndarray, Node, Function)) and not isinstance(val, Node):
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = cache.get(self.attrname, _not_found_)
                if not isinstance(val, (int, str, np.ndarray, Node, Function)) and not isinstance(val, Node):
                    val = self.func(instance)

                    try:
                        cache.put(val, self.attrname)
                    except TypeError as error:
                        msg = (
                            f"The '_entry' attribute on {type(instance).__name__!r} instance "
                            f"does not support item assignment for caching {self.attrname!r} property."
                        )
                        raise TypeError(error) from None
        return val

    def __set__(self, instance: Node, value):
        if not isinstance(instance, Node):
            raise NotImplemented

        instance._entry.put(value, self.attrname)
