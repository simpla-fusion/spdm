import collections.abc
import inspect
from _thread import RLock
from typing import Any, Callable, Generic, Type, TypeVar, Union, final, get_args

import numpy as np

from ..common.logger import logger
from ..common.tags import _not_found_, _undefined_
from .Entry import Entry
from .Node import Node

_TObject = TypeVar("_TObject")
_T = TypeVar("_T")


class sp_property(Generic[_TObject]):
    """return a sp_property attribute.

       用于辅助为Node定义property。
       - 在读取时将cache中的data转换为类型_TObject
       - 缓存property function,并缓存其输出

    Args:
        Generic ([type]): [description]
    """

    def __init__(self, getter=_undefined_, setter=_undefined_, deleter=_undefined_, doc=_undefined_, default_value=_undefined_, validator=_undefined_, **kwargs):
        self.lock = RLock()

        self.getter = getter
        self.setter = setter
        self.deleter = deleter
        self.__doc__ = doc

        self.property_cache_key = getter if not callable(getter) else _undefined_
        self.property_name = _undefined_
        self.property_type = _undefined_

        self.default_value = default_value
        self.validator = validator
        self.opts = kwargs

    def __set_name__(self, owner, name):

        self.property_name = name

        if self.__doc__ is not _undefined_:
            pass
        elif callable(self.getter):
            self.__doc__ = self.getter.__doc__
        else:
            self.__doc__ = f"property:{self.property_name}"

        if self.property_cache_key is _undefined_:
            self.property_cache_key = name

        if self.property_type is _undefined_ and inspect.isfunction(self.getter):
            self.property_type = self.getter.__annotations__.get("return", _undefined_)
        else:
            #  @ref: https://stackoverflow.com/questions/48572831/how-to-access-the-type-arguments-of-typing-generic?noredirect=1
            orig_class = getattr(self, "__orig_class__", None)
            if orig_class is not None:
                child_cls = get_args(self.__orig_class__)
                if child_cls is not None and len(child_cls) > 0 and inspect.isclass(child_cls[0]):
                    child_cls = child_cls[0]
            self.property_type = child_cls

        if self.property_name != self.property_cache_key:
            logger.warning(
                f"The attribute name '{self.property_name}' is different from the cache '{self.property_cache_key}''.")

    def validate(self, value):

        if self.validator is not _undefined_:
            return self.validator(value,  self.property_type)
        else:
            orig_class = getattr(value, "__orig_class__", value.__class__)

            return self.property_type in [None, _undefined_]  \
                or orig_class == self.property_type \
                or (inspect.isclass(orig_class)
                    and inspect.isclass(self.property_type)
                    and issubclass(orig_class, self.property_type))

    def __set__(self, instance: Node, value: Any):
        if not isinstance(instance, Node):
            raise TypeError(type(instance))

        with self.lock:
            if callable(self.setter):
                self.setter(value)

            instance._entry.put(self.property_cache_key,  value)

    def __get__(self, instance: Node, owner=None) -> _T:
        if not isinstance(instance, Node):
            raise TypeError(type(instance))

        if self.property_name is None:
            logger.warning("Cannot use sp_property instance without calling __set_name__ on it.")

        with self.lock:
            value = instance._entry.get(self.property_cache_key, _not_found_)

            if not self.validate(value):
                if callable(self.getter):
                    value = self.getter(instance, **self.opts)
                elif value is _not_found_:
                    value = self.default_value

                value = instance.update_child(value, self.property_cache_key, type_hint=self.property_type, **self.opts)

        return value

    def __delete__(self, instance: Node) -> None:
        if not isinstance(instance, Node):
            raise TypeError(type(instance))

        with self.lock:
            if callable(self.deleter):
                self.deleter()
            instance._entry.child(self.property_cache_key).erase()
