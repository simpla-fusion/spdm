import collections
import collections.abc
import dataclasses
import inspect
from _thread import RLock
from functools import cached_property
from typing import Any, Callable, Generic, TypeVar, Union, final, get_args

import numpy as np

from ..common.logger import logger
from ..common.tags import _not_found_, _undefined_
from .Entry import Entry
from .Node import Node

_TObject = TypeVar("_TObject")
_T = TypeVar("_T")


class _sp_property(Generic[_TObject]):

    def __init__(self, func: Callable[..., _TObject]):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.lock = RLock()
        self.return_type = func.__annotations__.get("return", None)

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def _check_type(self, value):
        orig_class = getattr(value, "__orig_class__", value.__class__)
        return self.return_type is None \
            or orig_class == self.return_type \
            or (inspect.isclass(orig_class)
                and inspect.isclass(self.return_type)
                and issubclass(orig_class, self.return_type))

    def _convert(self, value: _T, parent=None) -> _T:
        # if self._check_type(value):
        #     n_value = value
        # else:
        #     n_value = self.return_type(value)
        if inspect.isclass(self.return_type) and issubclass(self.return_type, Node):
            return self.return_type(value, parent=parent)
        # elif hasattr(parent, "_convert"):
        #     return parent._convert(value, parent=parent, attribute=self.return_type)
        else:
            return self.return_type(value)

    def _get_entry(self, instance: Node) -> Entry:
        try:
            entry = getattr(instance, "_entry", _not_found_)
            if entry is _not_found_:
                entry = Entry(instance.__dict__)
        except AttributeError as error:
            logger.exception(error)
            raise AttributeError(error)

        return entry

    def __set__(self, instance: Node, value: Any):
        if instance is None:
            return self
        with self.lock:
            if self._check_type(value):
                if value._parent is None:
                    value._parent = instance
                else:
                    value = value._duplicate(parent=instance)
                self._get_entry(instance).put(self.attrname, value)
            else:
                self._get_entry(instance).put(self.attrname, self._convert(value, instance))

    def __get__(self, instance: Node, owner=None) -> _T:
        if instance is None:
            return self

        if self.attrname is None:
            raise TypeError("Cannot use sp_property instance without calling __set_name__ on it.")

        with self.lock:
            entry = self._get_entry(instance)

            value = entry.get(self.attrname, _not_found_)

            if not self._check_type(value):
                n_value = self._convert(self.func(instance), instance)

                entry.put(self.attrname, n_value)
            else:
                n_value = value

            return n_value

        return n_value

    def __delete__(self, instance: Node) -> None:
        if instance is None:
            return
        with self.lock:
            self._get_entry(instance).child(self.attrname).erase()


def sp_property(func: Callable[..., _TObject]) -> _sp_property[_TObject]:
    return _sp_property[_TObject](func)
