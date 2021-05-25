import collections.abc
import logging
import typing
from _thread import RLock
from typing import (Callable, Generic, Mapping, Sequence, TypeVar, Union,
                    get_args)

from ..util.logger import logger
from ..util.utilities import _not_found_
from .Node import Node, _TObject

_TObject = TypeVar('_TObject')


class _SpProperty(Generic[_TObject]):
    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.lock = RLock()
        self._return_type = func.__annotations__.get("return", type(None))

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
            raise TypeError("Cannot use _SpProperty instance without calling __set_name__ on it.")

        val = instance._entry.get(self.attrname, _not_found_)
        if val is _not_found_ or not (self._return_type is None or isinstance(val, self._return_type)):
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = instance._entry.get(self.attrname, _not_found_)
                if val is _not_found_ or not (self._return_type is None or isinstance(val, self._return_type)):
                    try:
                        val = self._return_type(self.func(instance), parent=instance)
                    except Exception as error:
                        logger.error(f"Can not create {self._obj_type.__name__}")
                        raise error from None

                    try:
                        instance._entry.put(val, self.attrname)
                    except TypeError as error:
                        logger.error(f"Can not put value to {self.attrname}")
                        raise TypeError(error) from None
        return val

    def __set__(self, instance: Node, value):
        if not isinstance(instance, Node):
            raise NotImplemented
        with self.lock:
            try:
                instance._entry.put(value, self.attrname)
            except TypeError as error:
                logger.error(f"Can not put value to {self.attrname}")
                raise TypeError(error) from None


def sp_property(func: Callable[..., _TObject]) -> _SpProperty[_TObject]:
    return _SpProperty[_TObject](func)


# def sp_property_with_parameter(*args, **kwargs) -> Callable[[Callable[..., _TObject]], _SpProperty[_TObject]]:
#     """
#         NOTE: Pylance failed!
#     """
#     def _wrapper(wrapped: Callable[..., _TObject], _args=args, _kwargs=kwargs) -> _SpProperty[_TObject]:
#         return _SpProperty[_TObject](wrapped)

#     return _wrapper
