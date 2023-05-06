"""
定义一个property, 要求其所在的class必须有一个_as_child方法，用于将其转换为type_hint 指定的类型。
    ```python
        class Foo(Dict):
            pass

        class Doo(Dict):

            f0 = sp_property(type_hint=Foo)      # 优先级最高, 不兼容IDE的类型提示

            f1: Foo = sp_property()              # 推荐，可以兼容IDE的类型提示

            ######################################################
            @sp_property     
            def f3(self) -> Foo:                 # 用于定义f3的getter操作，与@property.getter类似  
                'This is  f3!'
                return self.get("f3", {})

            @f3.setter
            def f3(self,value)->None:            # 功能与@property.setter  类似, NOT IMPLEMENTED YET!!
                self._entry.put("f3",value)

            @f3.deleter
            def f3(self)->None:                  # 功能与@property.deleter 类似, NOT IMPLEMENTED YET!!
                self._entry.child("f3").erase()
            ######################################################
                                                 # 完整版本
            def get_f4(self,default={})->Foo:
                return self.get("f4", default)

            def set_f4(self,value)->None:
                return self.set("f4", value) 

            def del_f4(self,value)->None:
                return self.set("f4", value) 

            f4 = sp_property(get_f4,set_f4,del_f4,"I'm f4",type_hint=Foo)
        ```

"""

from __future__ import annotations

import collections
import collections.abc
import inspect
import typing
from _thread import RLock
from typing import Any

import numpy as np

from spdm.data.Node import Node

from ..utils.logger import logger
from ..utils.tags import _not_found_, _undefined_
from .Container import Container

_T = typing.TypeVar("_T")


class SpPropertyClass(Container, typing.MutableMapping[str, typing.Any]):
    def __init__(self, *args, cache=None,  **kwargs) -> None:
        super().__init__(*args, cache=cache if cache is not None else {},   **kwargs)

    def _type_hint(self, key: str) -> typing.Type:
        return typing.get_type_hints(self.__class__).get(key, None)\
            or getattr(getattr(self.__class__, key, None), "type_hint", None)


class sp_property(typing.Generic[_T]):  # type: ignore
    def __init__(self,
                 getter: typing.Callable[[typing.Any], typing.Any] = None,
                 setter=None,
                 deleter=None,
                 type_hint: typing.Type = None,
                 doc: typing.Optional[str] = None,
                 strict: bool = True,
                 **kwargs):

        self.lock = RLock()

        self.getter = getter
        self.setter = setter
        self.deleter = deleter
        if doc is not None:
            self.__doc__ = doc

        self.property_cache_key = getter if not callable(getter) else None
        self.property_name: str = None
        self.type_hint = type_hint
        self.strict = strict
        self.appinfo = kwargs

    def __set_name__(self, owner, name):
        # TODO：
        #    若 owner 是继承自具有属性name的父类，则默认延用父类sp_property的设置

        self.property_name = name
        self.appinfo.setdefault("name", name)
        if self.__doc__ is not None:
            pass
        elif callable(self.getter):
            self.__doc__ = self.getter.__doc__
        else:
            self.__doc__ = f"sp_roperty:{self.property_name}"

        if self.property_cache_key is None:
            self.property_cache_key = name

        if self.property_name != self.property_cache_key:
            logger.warning(
                f"The property name '{self.property_name}' is different from the cache '{self.property_cache_key}''.")

    def _get_type_hint(self, owner_cls):

        if self.type_hint is not None:
            return self.type_hint
        else:
            t_hints = typing.get_type_hints(owner_cls)
            self.type_hint = t_hints.get(self.property_name, None)

        if self.type_hint is None:
            #  @ref: https://stackoverflow.com/questions/48572831/how-to-access-the-type-arguments-of-typing-generic?noredirect=1
            orig_class = getattr(self, "__orig_class__", None)
            if orig_class is not None:
                child_cls = typing.get_args(self.__orig_class__)
                if child_cls is not None and len(child_cls) > 0 and inspect.isclass(child_cls[0]):
                    self.type_hint = child_cls[0]

        if self.type_hint is None and inspect.isfunction(self.getter):
            self.type_hint = self.getter.__annotations__.get("return", None)

        if len(self.appinfo) == 0 and not isinstance(self.appinfo, collections.ChainMap):
            for base in owner_cls.__bases__:
                if isinstance(getattr(base, self.property_name, None), sp_property):
                    self.appinfo.update(getattr(base, self.property_name).appinfo)

    def __set__(self, instance: SpPropertyClass, value: typing.Any):
        assert (instance is not None)

        self._get_type_hint(instance.__class__)

        if self.property_name is None or self.property_cache_key is None:
            logger.warning("Cannot use sp_property instance without calling __set_name__ on it.")

        with self.lock:
            if callable(self.setter):
                self.setter(instance, value)
            else:
                instance._as_child(key=self.property_cache_key, value=value,
                                   type_hint=self.type_hint, appinfo=self.appinfo)

    def __get__(self, instance: SpPropertyClass | None, owner=None) -> _T | sp_property[_T]:
        if instance is None:
            # 当调用 getter(cls, <name>) 时执行
            return self
        elif not isinstance(instance, SpPropertyClass):
            raise TypeError(f"Class '{instance.__class__.__name__}' must be a subclass of 'SpPropertyClass'.")

        # 当调用 getter(obj, <name>) 时执行
        self._get_type_hint(owner)

        if self.property_name is None or self.property_cache_key is None:
            logger.warning("Cannot use sp_property instance without calling __set_name__ on it.")

        with self.lock:
            if isinstance(instance._cache, collections.abc.Mapping):
                value = instance._cache.get(self.property_cache_key, _not_found_)
            else:
                value = _not_found_

            if value is _not_found_ and callable(self.getter):
                value = self.getter(instance)

            value = instance._as_child(key=self.property_cache_key,
                                       value=value,
                                       type_hint=self.type_hint,
                                       strict=self.strict,
                                       ** self.appinfo)

            if self.strict and value is _not_found_:
                raise AttributeError(
                    f"The value of property '{owner.__name__ if owner is not None else 'none'}.{self.property_name}' is not assigned!")

        return value

    def __delete__(self, instance: typing.Any) -> None:
        with self.lock:
            if callable(self.deleter):
                self.deleter(instance)
            elif isinstance(instance._cache, collections.abc.MutableMapping):
                if self.property_cache_key in instance._cache:
                    del instance._cache[self.property_cache_key]
            else:
                raise AttributeError(f"Cannot delete property '{self.property_name}'")
                # del instance._cache[self.property_cache_key]

    def __call__(self, func, *args: Any, **kwds: Any) -> Any:
        """ 用于装饰函数，将函数的返回值作为属性值返回
        """
        self.getter = func
        return self
