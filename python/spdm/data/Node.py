import bisect
import collections
import collections.abc
import dataclasses
import enum
import functools
import inspect
import logging
import pprint
from _thread import RLock
from enum import IntFlag
from functools import cached_property
from typing import (Any, Callable, Generic, Iterator, Mapping, MutableMapping,
                    MutableSequence, Optional, Sequence, Tuple, Type, TypeVar,
                    Union, final, get_args)

from numpy.lib.arraysetops import isin
from ..util.sp_export import sp_find_module
from ..numlib import np, scipy
from ..util.logger import logger
from ..util.utilities import _not_found_, _undefined_, serialize
from .Entry import (_DICT_TYPE_, _LIST_TYPE_, Entry, EntryCombiner,
                    EntryContainer, _next_, _TKey, _TObject, _TQuery,
                    as_dataclass)

_TNode = TypeVar('_TNode', bound='Node')

_T = TypeVar("_T")


class Node(EntryContainer[_TObject]):
    r"""
        @startuml

        class Node{
            name    : String
            parent  : Node
            value   : Group or Data
        }

        class Group{
            children : Node[*]
        }

        Node *--  Node  : parent

        Node o--  Group : value
        Node o--  Data  : value

        Group *-- "*" Node

        @enduml
    """
    __slots__ = "_parent",   "__orig_class__", "__new_child__"

    def __init__(self, cache: Any = None, *args, parent=None, new_child=_undefined_,   **kwargs):
        super().__init__(cache)
        self._parent = parent
        self.__new_child__ = new_child

    def __repr__(self) -> str:
        return f"<{getattr(self,'__orig_class__',self.__class__.__name__)} id='{self.nid}' />"

    @property
    def nid(self) -> str:
        id = self.get("@id", "")
        if not not id:
            id = f"id='{id}'"

    def __serialize__(self) -> Any:
        return serialize(self.get(Entry.ops.dump))

    @classmethod
    def __deserialize__(cls, desc: Any) -> _TNode:
        module_path = getattr(cls, "_module_prefix", "") + desc.get("@ids", "")

        if not module_path:
            new_cls = cls
        else:
            new_cls = sp_find_module(module_path)

        if not issubclass(new_cls, Node):
            raise TypeError(f"{new_cls.__name__} is not a 'Node'!")

        obj: Node = object.__new__(new_cls)
        obj.put(Entry.ops.assign, desc)
        return obj

    def __duplicate__(self) -> _TNode:
        obj = super().__duplicate__()
        obj._parent = self._parent
        obj.__new_child__ = self.__new_child__
        return obj

    @property
    def __parent__(self) -> object:
        return self._parent

    def __hash__(self) -> int:
        return NotImplemented

    @property
    def empty(self) -> bool:
        return self.get(Entry.ops.exists)

    """
        @startuml
        [*] --> Empty
        Empty       --> Sequence        : as_sequence, __update__(list), __setitem__(int,v),__getitem__(int)
        Empty       --> Mapping         : as_mapping , __update__(dict), __setitem__(str,v),__getitem__(str)
        Empty       --> Empty           : clear


        Item        --> Item            : "__fetch__"
        Item        --> Empty           : clear
        Item        --> Sequence        : __setitem__(_next_,v),__getitem__(_next_),as_sequence
        Item        --> Illegal         : as_mapping

        Sequence    --> Empty           : clear
        Sequence    --> Sequence        : as_sequence
        Sequence    --> Illegal         : as_mapping

        Mapping     --> Empty           : clear
        Mapping     --> Mapping         : as_mapping
        Mapping     --> Sequence        :  __setitem__(_next_,v),__getitem__(_next_),as_sequence


        Illegal     --> [*]             : Error

        @enduml
    """

    def __pre_process__(self, value: Any, *args, **kwargs) -> Any:
        return value

    def __post_process__(self, value: _T,   /, *args, parent=None,   **kwargs) -> Union[_T, _TNode]:
        if not isinstance(value,  Entry):
            return value

        if self.__new_child__ is _undefined_:
            child_cls = None
            #  @ref: https://stackoverflow.com/questions/48572831/how-to-access-the-type-arguments-of-typing-generic?noredirect=1
            orig_class = getattr(self, "__orig_class__", None)
            if orig_class is not None:
                child_cls = get_args(self.__orig_class__)
                if child_cls is not None and len(child_cls) > 0 and inspect.isclass(child_cls[0]):
                    child_cls = child_cls[0]

            self.__new_child__ = child_cls

        parent = parent if parent is not None else self
        if inspect.isclass(self.__new_child__):
            if isinstance(value, self.__new_child__):
                pass
            elif issubclass(self.__new_child__, Node):
                value = self.__new_child__(value, *args, parent=parent, **kwargs)
            else:
                value = self.__new_child__(value, *args,  **kwargs)

        elif callable(self.__new_child__):
            value = self.__new_child__(value, *args, parent=parent,  **kwargs)
        elif self.__new_child__ is not None:
            raise TypeError(f"Illegal type! {self.__new_child__}")
        elif value.is_list:
            value = List(value, *args, parent=parent, new_child=self.__new_child__,  **kwargs)
        elif value.is_dict:
            value = Dict(value, *args, parent=parent, new_child=self.__new_child__,  **kwargs)
        else:
            value = Node(value, *args, parent=parent, new_child=self.__new_child__, **kwargs)

        return value

    def __setitem__(self, query: _TQuery, value: _T) -> _T:
        return self.put([query, Entry.ops.assign], value)

    def __getitem__(self, query: _TQuery) -> _TNode:
        return self.get(query)

    def __delitem__(self, query: _TQuery) -> bool:
        _, status = self.put([query, Entry.ops.erase])
        return status

    def __contains__(self, query: _TQuery) -> bool:
        return self.get([query, Entry.ops.exists])

    def __len__(self) -> int:
        return self.get(Entry.ops.count)

    def __iter__(self) -> Iterator[_T]:
        for obj in self._entry.iter():
            yield self.__post_process__(obj)

    def __eq__(self, other) -> bool:
        val, _ = self.put(Entry.ops.equal, other)
        return val

    def __bool__(self) -> bool:
        return not self.empty  # and (not self.__fetch__())

    class Category(IntFlag):
        UNKNOWN = 0
        ITEM = 0x000
        DICT = 0x100
        LIST = 0x200
        ENTRY = 0x400
        ARRAY = 0x010
        INT = 0x001
        FLOAT = 0x002
        COMPLEX = 0x004
        STRING = 0x008

    @staticmethod
    def __type_category__(d) -> IntFlag:
        flag = Node.Category.UNKNOWN
        if hasattr(d,  "__array__"):
            flag |= Node.Category.ARRAY
            # if np.issubdtype(d.dtype, np.int64):
            #     flag |= Node.Category.INT
            # elif np.issubdtype(d.dtype, np.float64):
            #     flag |= Node.Category.FLOAT
        elif isinstance(d, collections.abc.Mapping):
            flag |= Node.Category.DICT
        elif isinstance(d, collections.abc.Sequence):
            flag |= Node.Category.LIST
        elif isinstance(d, int):
            flag |= Node.Category.INT
        elif isinstance(d, float):
            flag |= Node.Category.FLOAT
        elif isinstance(d, str):
            flag |= Node.Category.STRING
        # if isinstance(d, (Entry)):
        #     flag |= Node.Category.ENTRY

        return flag

    @property
    def __category__(self) -> Category:
        return Node.__type_category__(self._entry)


class List(Node[_T], Sequence[_T]):
    __slots__ = ("_v_args", "_combine")

    def __init__(self, cache: Optional[Sequence] = None, *args, parent=None, default_value_when_combine=None,  **kwargs) -> None:
        Node.__init__(self, cache, *args, parent=parent, **kwargs)
        self._v_args = (args, kwargs)
        self._combine = default_value_when_combine

    def __serialize__(self) -> Sequence:
        return [serialize(v) for v in self._as_list()]

    @property
    def __category__(self):
        return super().__category__ | Node.Category.LIST

    def __len__(self) -> int:
        return super().__len__()

    def __setitem__(self, query: _TQuery, v: _T) -> None:
        super().__setitem__(query, v)

    def __getitem__(self, query: _TQuery) -> _T:
        obj = self.get(query)
        if isinstance(obj, (collections.abc.Sequence)) and not isinstance(obj, str):
            if len(obj) > 1:
                obj = Entry(obj)
            else:
                obj = obj[0]
        return self.__post_process__(obj, parent=self._parent)

    def __delitem__(self, query: _TQuery) -> None:
        super().__delitem__(query)

    def __iter__(self) -> Iterator[_T]:
        for idx in range(len(self)):
            yield self.__getitem__(idx)

    def __iadd__(self, other):
        self.put(_next_, other)
        return self

    def sort(self):
        if hasattr(self._entry.__class__, "sort"):
            self._entry.sort()
        else:
            raise NotImplementedError()

    @property
    def combine(self) -> _T:
        if not isinstance(self._combine, Node):
            if self._entry.__class__ is not Entry:
                raise NotImplementedError(type(self._entry))
            self._combine = self.__post_process__(
                EntryCombiner(self._entry._cache,
                              path=self._entry._path,
                              default_value=self._combine),
                parent=self._parent, not_insert=True)
        return self._combine

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        for element in self.__iter__():
            if hasattr(element.__class__, 'update'):
                element.update(*args, **kwargs)

    def reset(self, value=None):
        if isinstance(value, (collections.abc.Sequence)):
            super().reset(value)
        else:
            self._combine = value
            super().reset()


class Dict(Node[_T], Mapping[str, _T]):
    __slots__ = ()

    def __init__(self, cache: Optional[Mapping] = None, *args,  **kwargs):
        Node.__init__(self, cache, *args, **kwargs)

    @property
    def __category__(self):
        return super().__category__ | Node.Category.DICT

    def __getitem__(self, query: _TKey) -> _TObject:
        return super().__getitem__(query)

    def __setitem__(self, query: _TKey, value: _T) -> None:
        super().__setitem__(query, value)

    def __delitem__(self, query: _TKey) -> None:
        super().__delitem__(query)

    def __iter__(self) -> Iterator[str]:
        yield from self.keys()

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o)

    def __contains__(self, o: object) -> bool:
        return super().__contains__(o)

    def __ior__(self, other):
        return self.put(Entry.ops.update, other)

    def items(self) -> Iterator[Tuple[str, _T]]:
        yield from super().items()

    def keys(self) -> Iterator[str]:
        yield from super().keys()

    def values(self) -> Iterator[_T]:
        yield super().items()

    # def _as_dict(self) -> Mapping:
    #     cls = self.__class__
    #     if cls is Dict:
    #         return self._entry._data
    #     else:
    #         properties = set([k for k in self.__dir__() if not k.startswith('_')])
    #         res = {}
    #         for k in properties:
    #             prop = getattr(cls, k, None)
    #             if inspect.isfunction(prop) or inspect.isclass(prop) or inspect.ismethod(prop):
    #                 continue
    #             elif isinstance(prop, cached_property):
    #                 v = prop.__get__(self)
    #             elif isinstance(prop, property):
    #                 v = prop.fget(self)
    #             else:
    #                 v = getattr(self, k, _not_found_)
    #             if v is _not_found_:
    #                 v = self._entry.find(k)
    #             if v is _not_found_ or isinstance(v, Entry):
    #                 continue
    #             # elif hasattr(v, "__serialize__"):
    #             #     res[k] = v.__serialize__()
    #             # else:
    #             #     res[k] = serialize(v)
    #             res[k] = v
    #         return res
    # self.__reset__(d.keys())
    # def __reset__(self, d=None) -> None:
    #     if isinstance(d, str):
    #         return self.__reset__([d])
    #     elif d is None:
    #         return self.__reset__([d for k in dir(self) if not k.startswith("_")])
    #     elif isinstance(d, Mapping):
    #         properties = getattr(self.__class__, '_properties_', _not_found_)
    #         if properties is not _not_found_:
    #             data = {k: v for k, v in d.items() if k in properties}
    #         self._entry = Entry(data, parent=self._entry.parent)
    #         self.__reset__(d.keys())
    #     elif isinstance(d, Sequence):
    #         for key in d:
    #             if isinstance(key, str) and hasattr(self, key) and isinstance(getattr(self.__class__, key, _not_found_), functools.cached_property):
    #                 delattr(self, key)


class _SpProperty(Generic[_T]):
    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.lock = RLock()
        self.return_type = func.__annotations__.get("return", None)

    def _isinstance(self, obj) -> bool:
        res = True
        if self.return_type is not None:
            orig_class = getattr(obj, "__orig_class__", obj.__class__)
            res = inspect.isclass(orig_class) \
                and inspect.isclass(self.return_type) \
                and issubclass(orig_class, self.return_type) \
                or orig_class == self.return_type

        return res

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def __put__(self, cache: Any, val: Any):
        if isinstance(val, Node):
            logger.debug((self.attrname, type(val._entry), type(cache), val._entry._data is cache._data))

        try:
            cache.insert(self.attrname, val)
        except TypeError as error:
            # logger.error(f"Can not put value to '{self.attrname}'")
            raise TypeError(error) from None

    def __get__(self, instance: Any, owner=None) -> _T:
        cache = getattr(instance, "_entry", Entry(instance.__dict__))

        if self.attrname is None:
            raise TypeError("Cannot use _SpProperty instance without calling __set_name__ on it.")
        # elif isinstance(cache, Entry) and not cache.writable:
        #     logger.error(f"Attribute cache is not writable!")
        #     raise AttributeError(self.attrname)

        val = cache.get(self.attrname, default_value=_not_found_, shallow=True)

        if not self._isinstance(val):
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = cache.get(self.attrname, default_value=_not_found_, shallow=True)
                # FIXME: Thread safety is not guaranteed! solution: lock on cache???
                if not self._isinstance(val):
                    obj = self.func(instance)
                    if not self._isinstance(obj) and getattr(instance, '__new_child__', None) not in (None, _not_found_, _undefined_):
                        obj = instance.__new_child__(obj)
                    if not self._isinstance(obj):
                        origin_type = getattr(self.return_type, '__origin__', self.return_type)
                        if dataclasses.is_dataclass(origin_type):
                            obj = as_dataclass(origin_type, obj)
                        elif inspect.isclass(origin_type) and issubclass(origin_type, Node):
                            obj = self.return_type(obj, parent=instance)
                        elif origin_type is np.ndarray:
                            obj = np.asarray(obj)
                        elif callable(self.return_type) is not None:
                            try:
                                tmp = self.return_type(obj)
                            except Exception as error:
                                logger.error(f"{self.attrname} {self.return_type} {type(obj)} : {error}")
                                raise error
                            else:
                                obj = tmp

                    if obj is not val and isinstance(cache, Entry) and cache.writable:
                        val = cache.insert(self.attrname, obj,  assign_if_exists=True)
                    else:
                        val = obj
        return val

    def __set__(self, instance: Any, value: Any):
        with self.lock:
            cache = getattr(instance, "_entry", Entry(instance.__dict__))
            cache.insert(self.attrname, value, assign_if_exists=True)

    # def __del__(self, instance: Any):
    #     with self.lock:
    #         cache = getattr(instance, "_entry", instance.__dict__)

    #         try:
    #             cache.delete(self.attrname)
    #         except Exception:
    #             try:
    #                 del cache[self.attrname]
    #             except TypeError as error:
    #                 logger.error(f"Can not delete '{self.attrname}'")
    #                 raise TypeError(error)


def sp_property(func: Callable[..., _T]) -> _SpProperty[_T]:
    return _SpProperty[_T](func)
