import inspect
import typing
from _thread import RLock

from ..common.tags import _undefined_
from ..util.logger import logger
from .Dict import Dict
from .Node import Node

_TObject = typing.TypeVar("_TObject")
_T = typing.TypeVar("_T")


class sp_property(typing.Generic[_TObject]):  # type: ignore
    """
    用于辅助为Node定义property。
       - 在读取时将cache中的data转换为类型_TObject
       - 缓存property function,并缓存其输出

       用法:


        ```python
        class Foo(Dict):
            def __init__(self, data: dict, *args, **kwargs):
                super().__init__(data, *args, **kwargs)


        class Doo(Dict):
            def __init__(self, *args,   **kwargs):
                super().__init__(*args,  **kwargs)

            f0 = sp_property(type_hint=Foo)      # 优先级最高, 不兼容IDE的类型提示

            f1: Foo = sp_property()              # 推荐，可以兼容IDE的类型提示

            f2 = sp_property[Foo]()              # 与1等价，可以兼容IDE的类型提示



            @sp_property[Foo]     
            def f3(self) :                       # 要求Python >=3.9  
                return self.get("f3", {})

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

        Args:
        Generic ([type]): [description]
    """

    def __init__(self,
                 getter=None,
                 setter=None,
                 deleter=None,
                 default_value=None,
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

    def __set_name__(self, owner, name):
        if not issubclass(owner, Dict):
            raise RuntimeError(f"'sp_property' only can be define as a property of Dict! [name={name} owner={owner}]")
        # elif not hasattr(owner, '_properties'):
        #     owner._properties = {name}
        # elif name in owner._properties:
        #     raise RuntimeError(f"The property {name} has already been defined! {owner}:{owner._properties} ")
        # else:
        #     owner._properties.add(name)

        self.property_name = name

        if self.__doc__ is not None:
            pass
        elif callable(self.getter):
            self.__doc__ = self.getter.__doc__
        else:
            self.__doc__ = f"property:{self.property_name}"

        if self.property_cache_key is None:
            self.property_cache_key = name

        if self.property_name != self.property_cache_key:
            logger.warning(
                f"The attribute name '{self.property_name}' is different from the cache '{self.property_cache_key}''.")

        if self.type_hint is None:
            self.type_hint = typing.get_type_hints(owner).get(name, None)

        if self.type_hint is None:
            #  @ref: https://stackoverflow.com/questions/48572831/how-to-access-the-type-arguments-of-typing-generic?noredirect=1
            orig_class = getattr(self, "__orig_class__", None)
            if orig_class is not None:
                child_cls = typing.get_args(self.__orig_class__)
                if child_cls is not None and len(child_cls) > 0 and inspect.isclass(child_cls[-1]):
                    self.type_hint = child_cls[-1]

        if self.type_hint is None and inspect.isfunction(self.getter):
            self.type_hint = self.getter.__annotations__.get("return", None)

    def __set__(self, instance: Node, value: typing.Any):
        if not isinstance(instance, Node):
            raise TypeError(type(instance))

        with self.lock:
            if callable(self.setter):
                self.setter(instance, value)
            else:
                instance._entry.child(self.property_cache_key).insert(value)

    def __get__(self, instance: Dict, owner=None) -> typing.Optional[_TObject]:
        if not isinstance(instance, Dict):
            if instance is None:
                return None
            else:
                raise RuntimeError(
                    f"sp_property is only valid for 'Node', not for {type(instance)} '{self.property_name}'.")
            # return {}

        if self.property_name is None or self.property_cache_key is None:
            logger.warning("Cannot use sp_property instance without calling __set_name__ on it.")

        with self.lock:
            value = instance._as_child(key=str(self.property_cache_key),
                                       type_hint=self.type_hint,
                                       default_value=self.default_value,
                                       getter=self.getter,
                                       **self.kwargs)

        return value

    def __delete__(self, instance: Node) -> None:
        if not isinstance(instance, Node):
            raise TypeError(type(instance))

        with self.lock:
            if callable(self.deleter):
                self.deleter(instance)
            else:
                instance._entry.child(self.property_cache_key).remove()
