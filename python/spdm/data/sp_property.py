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
import pprint
import inspect
import typing
from copy import deepcopy
from _thread import RLock
from enum import Enum
from ..utils.envs import SP_DEBUG
from ..utils.logger import logger
from ..utils.tags import _not_found_
from .Entry import Entry
from .Function import Function
from .HTree import HTree, Dict
from ..utils.tree_utils import merge_tree_recursive


class SpTree(Dict):
    """  支持 sp_property 的 Dict  """

    def __init__(self, *args, **kwargs) -> None: super().__init__(*args, **kwargs)

    def __get_property__(self, key: str, *args, **kwargs) -> SpTree: return self._get(key, *args, **kwargs)

    def __set_property__(self, key: str,  value: typing.Any = None, **kwargs) -> None: self.update(key, value)

    def __del_property__(self, key: str, **kwargs): self._remove(key)

    def dump(self, entry: Entry | None = None, force=False, quiet=True) -> Entry:
        if entry is None:
            entry = Entry({})
            force = True

        for k, _ in inspect.getmembers(self.__class__, is_sp_property):

            try:
                prop = getattr(self, k, None)
                if prop is _not_found_:
                    prop = None
                elif isinstance(prop, Function):
                    prop = prop.__array__()

            except Exception as error:
                if SP_DEBUG == "CRITICAL":
                    raise RuntimeError(f"Fail to dump property: {self.__class__.__name__}.{k}") from error
                else:
                    logger.warning(f"Fail to dump property: {self.__class__.__name__}.{k}")
            else:

                if isinstance(prop, Enum):
                    prop = {"name": prop.name, "index": prop.value}

                if isinstance(prop, HTree):
                    prop.dump(entry.child(k), quiet=quiet)
                else:
                    entry.child(k).insert(prop)
        if force:
            return entry._data
        else:
            return entry


class AttributeTree(SpTree):

    def __getattr__(self, key: str, *args, **kwargs):
        if key.startswith("__"):
            return super().__getattribute__(key)
        else:
            return self.__get_property__(key, *args, _type_hint=AttributeTree | None,  **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self.__get_property__(*args, _type_hint=AttributeTree | None,  **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self.__set_property__(*args,  **kwargs)

    def __iter__(self) -> typing.Generator[_T, None, None]:
        """ 遍历 children """
        for v in self.children():
            yield v

    def dump(self, entry: Entry | None = None, force=False, quiet=True) -> Entry:
        if entry is None:
            return deepcopy(self._cache)
        else:
            entry.update(self._cache)
            return entry


_T = typing.TypeVar("_T")


class sp_property(typing.Generic[_T]):
    """
    用于为 SpPropertyClass 类（及其子类）定义一个property, 并确保其类型为type_hint 指定的类型。

    例如：
    ``` python
        class Foo(SpPropertyClass):
            # 方法一
            @sp_property
            def a(self) -> float: return 128

            # 方法二
            @sp_property(coordinate1="../psi")
            def dphi_dpsi(self) -> Profile[float]: return self.a*2

            # 方法三
            phi: Profile[float] = sp_property(coordinate1="../psi")

    ```
    方法二、三中参数 coordinate1="../psi"，会在构建 Profile时传递给构造函数  Profile.__init__。

    方法三 会在创建class 是调用 __set_name__,
           会在读写property phi 时调用 __set__,__get__ 方法，
           从Node的_cache或entry获得名为 'phi' 的值，将其转换为 type_hint 指定的类型 Profile[float]。

    """

    def __init__(self,
                 getter: typing.Callable[[typing.Any], typing.Any] = None,
                 setter=None,
                 deleter=None,
                 type_hint: typing.Type = None,
                 doc: typing.Optional[str] = None,
                 strict: bool = False,
                 ** kwargs):
        """
            Parameters
            ----------
            getter : typing.Callable[[typing.Any], typing.Any]
                用于定义属性的getter操作，与@property.getter类似
            setter : typing.Callable[[typing.Any, typing.Any], None]
                用于定义属性的setter操作，与@property.setter类似
            deleter : typing.Callable[[typing.Any], None]
                用于定义属性的deleter操作，与@property.deleter类似
            type_hint : typing.Type
                用于指定属性的类型
            doc : typing.Optional[str]
                用于指定属性的文档字符串
            strict : bool
                用于指定是否严格检查属性的值是否已经被赋值
            metadata : typing.Any
                用于传递给构建  Node.__init__ 的参数


        """

        self.lock = RLock()

        self.getter = getter
        self.setter = setter
        self.deleter = deleter
        if doc is not None:
            self.__doc__ = doc

        self.property_name: str = None
        self.type_hint = type_hint
        self.strict = strict

        self.metadata = kwargs

        if isinstance(type_hint, str):
            raise RuntimeError(f"Invalid type_hint={type_hint}!")

    def __call__(self, func):
        """ 用于定义属性的getter操作，与@property.getter类似 """
        self.getter = func
        return self

    def __set_name__(self, owner, name):
        # TODO：
        #    若 owner 是继承自具有属性name的父类，则默认延用父类sp_property的设置

        self.property_name = name
        self.metadata.setdefault("name", name)
        if self.__doc__ is not None:
            pass
        elif callable(self.getter):
            self.__doc__ = self.getter.__doc__
        else:
            self.__doc__ = f"sp_roperty:{self.property_name}"

    def _get_desc(self, owner_cls, name: str = None, metadata: dict = None):

        if self.type_hint is not None:
            return self.type_hint, self.metadata

        type_hint = None

        if inspect.isfunction(self.getter):
            type_hint = typing.get_type_hints(self.getter).get("return", None)
        else:
            type_hint = typing.get_type_hints(owner_cls).get(name, None)

        if type_hint is None:
            #  @ref: https://stackoverflow.com/questions/48572831/how-to-access-the-type-arguments-of-typing-generic?noredirect=1
            orig_class = typing.get_origin(self.__class__)
            if orig_class is not None:
                child_cls = typing.get_args(orig_class)
                if child_cls is not None and len(child_cls) > 0 and inspect.isclass(child_cls[0]):
                    type_hint = child_cls[0]

        if not callable(type_hint):
            raise TypeError(type_hint)

        self.type_hint = type_hint

        metadata = self.metadata

        for base in owner_cls.__bases__:
            metadata = merge_tree_recursive(getattr(getattr(base, name, None), "_kwargs", None), metadata)

        return self.type_hint, metadata

    def __set__(self, instance:  SpTree[_T], value: typing.Any) -> None:
        assert (instance is not None)

        # type_hint, metadata = self._get_desc(instance.__class__, self.property_name, self.metadata)

        if self.property_name is None:
            logger.warning("Cannot use sp_property instance without calling __set_name__ on it.")

        with self.lock:
            instance.__set_property__(
                self.property_name,
                value=value,
                setter=self.setter)

    def __get__(self, instance:  SpTree[_T] | None, owner=None) -> _T:
        if instance is None:
            # 当调用 getter(cls, <name>) 时执行
            return self
        elif not isinstance(instance, SpTree):
            raise TypeError(f"Class '{instance.__class__.__name__}' must be a subclass of 'SpPropertyClass'.")

        # 当调用 getter(obj, <name>) 时执行

        type_hint, metdata = self._get_desc(owner, self.property_name)

        if self.property_name is None:
            logger.warning("Cannot use sp_property instance without calling __set_name__ on it.")

        with self.lock:
            value = instance.__get_property__(
                self.property_name,
                _type_hint=type_hint,
                _getter=self.getter,
                **metdata,
            )

            if self.strict and value is _not_found_:
                raise AttributeError(
                    f"The value of property '{owner.__name__ if owner is not None else 'none'}.{self.property_name}' is not assigned!")

        return value

    def __delete__(self, instance: SpTree[_T]) -> None:
        with self.lock:
            instance.__del_property__(self.property_name, deleter=self.deleter)


def is_sp_property(obj) -> bool: return isinstance(obj, sp_property)


def _process_sptree(cls,  **kwargs) -> typing.Type[SpTree]:
    if not inspect.isclass(cls):
        raise TypeError(f"Not a class {cls}")

    type_hints = typing.get_type_hints(cls)

    if not issubclass(cls, HTree):
        n_cls = type(f"_{cls.__name__}_s", (cls, SpTree), {"_metadata": getattr(cls, "_metadata", {})})
        n_cls.__module__ = cls.__module__
    else:
        n_cls = cls

    for _name, _type_hint in type_hints.items():
        prop = getattr(cls, _name, None)
        if isinstance(prop, sp_property):
            if _name in cls.__dict__:
                prop.type_hint = _type_hint
            else:
                prop = sp_property(type_hint=_type_hint,
                                   getter=prop.getter,
                                   setter=prop.setter,
                                   deleter=prop.deleter,
                                   **prop.metadata)
        else:
            prop = sp_property(type_hint=_type_hint, default_value=prop)

        prop.property_name = _name
        setattr(n_cls, _name, prop)

    setattr(n_cls, "_metadata", merge_tree_recursive(getattr(cls, "_metadata", None), kwargs))

    return n_cls


def sp_tree(cls: _T = None, /,   **kwargs) -> _T:
    # 装饰器，将一个类转换为 SpTree 类

    def wrap(cls):
        return _process_sptree(cls,  **kwargs)

    if cls is None:
        return wrap

    return wrap(cls)
