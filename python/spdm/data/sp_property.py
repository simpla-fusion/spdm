import collections.abc
from _thread import RLock
from typing import Generic, Mapping, Sequence

from typing import Generic
from ..util.utilities import _not_found_
from .Node import Node, _TObject


class sp_property(Generic[_TObject]):
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

        val = instance._entry.get(self.attrname, _not_found_)
        # if not isinstance(val, (int, str, np.ndarray, Function, Node))
        if isinstance(val, (collections.abc.Sequence, collections.abc.Mapping)) and not isinstance(val, (str, Node)):
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = instance._entry.get(self.attrname, _not_found_)
                if isinstance(val, (collections.abc.Sequence, collections.abc.Mapping)) and not isinstance(val, (str, Node)):
                    val = self.func(instance)

                    try:
                        instance._entry.put(val, self.attrname)
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
        with self.lock:
            instance._entry.put(value, self.attrname)
