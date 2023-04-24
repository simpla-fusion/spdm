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

import inspect
import typing
from _thread import RLock

import numpy as np

from ..common.tags import _not_found_, _undefined_
from ..util.logger import logger

_TObject = typing.TypeVar("_TObject")
_T = typing.TypeVar("_T")


class sp_property(typing.Generic[_TObject]):  # type: ignore
    def __init__(self,
                 getter=None,
                 setter=None,
                 deleter=None,
                 default_value=_not_found_,
                 type_hint=None,
                 doc: typing.Optional[str] = None,
                 force=False,
                 **kwargs):

        self.lock = RLock()

        self.getter = getter
        self.setter = setter
        self.deleter = deleter
        if doc is not None:
            self.__doc__ = doc

        self.property_cache_key = getter if not callable(getter) else None
        self.property_name = None
        self.type_hint = type_hint

        self.default_value = default_value
        self.kwargs = kwargs

    def __assert_obj_type(self, owner) -> None:
        if not inspect.isfunction(getattr(owner, "_as_child", None)):
            raise TypeError(
                f"sp_property is only valid for class with method '_as_child', not for {type(owner)} '{self.property_name}'.")

    def __set_name__(self, owner, name):
        self.__assert_obj_type(owner)

        # TODO：
        #    若 owner 是继承自具有属性name的父类，则默认延用父类sp_property的设置

        self.property_name = name

        if self.__doc__ is not None:
            pass
        elif callable(self.getter):
            self.__doc__ = self.getter.__doc__
        else:
            self.__doc__ = f"sp_roperty:{self.property_name}"

        if self.property_cache_key is None:
            self.property_cache_key = name

        if self.property_name != self.property_cache_key:
            logger.warning(f"The property name '{self.property_name}' is different from the cache '{self.property_cache_key}''.")

        if self.type_hint is None:
            self.type_hint = typing.get_type_hints(owner).get(name, None)

        if self.type_hint is None:
            #  @ref: https://stackoverflow.com/questions/48572831/how-to-access-the-type-arguments-of-typing-generic?noredirect=1
            orig_class = getattr(self, "__orig_class__", None)
            if orig_class is not None:
                child_cls = typing.get_args(self.__orig_class__)
                if child_cls is not None and len(child_cls) > 0 and inspect.isclass(child_cls[0]):
                    self.type_hint = child_cls[0]

        if self.type_hint is None and inspect.isfunction(self.getter):
            self.type_hint = self.getter.__annotations__.get("return", None)

    def __set__(self, instance: typing.Any, value: typing.Any):
        assert(instance is not None)
        self.__assert_obj_type(instance.__class__)

        if self.property_name is None or self.property_cache_key is None:
            logger.warning("Cannot use sp_property instance without calling __set_name__ on it.")

        with self.lock:
            if callable(self.setter):
                self.setter(instance, value)
            else:
                instance._as_child(self.property_cache_key, value)

    def __get__(self, instance: typing.Any, owner=None) -> typing.Union[sp_property[_TObject], _TObject]:
        if instance is None:
            # 当调用 getter(cls, <name>) 时执行
            return self

        # 当调用 getter(obj, <name>) 时执行
        self.__assert_obj_type(owner)
        if self.property_name is None or self.property_cache_key is None:
            logger.warning("Cannot use sp_property instance without calling __set_name__ on it.")

        with self.lock:
            value = instance._as_child(key=str(self.property_cache_key),
                                       type_hint=self.type_hint,
                                       default_value=self.default_value,
                                       getter=self.getter,
                                       **self.kwargs)
            if value is _not_found_:
                raise AttributeError(f"The value of property '{owner.__name__}.{self.property_name}' is not assigned!")

        return value

    def __delete__(self, instance: typing.Any) -> None:
        self.__assert_obj_type(instance)

        with self.lock:
            if callable(self.deleter):
                self.deleter(instance)
            else:
                raise AttributeError(f"Cannot delete property '{self.property_name}'")
                # del instance._cache[self.property_cache_key]
