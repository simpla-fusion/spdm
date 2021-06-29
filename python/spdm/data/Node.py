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
import dataclasses

from numpy.lib.arraysetops import _isin_dispatcher, isin
from ..util.sp_export import sp_find_module
from ..numlib import np, scipy
from ..util.logger import logger
from ..util.utilities import Tags, _not_found_, _undefined_, serialize
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
    __slots__ = "_parent",   "__orig_class__"

    def __init__(self, cache: Any = None, /, parent=None):
        super().__init__(cache)
        self._parent = parent

    def __repr__(self) -> str:
        return f"<{getattr(self,'__orig_class__',self.__class__.__name__)} id='{self.nid}' />"

    @property
    def nid(self) -> str:
        id = self.get("@id", "")
        if not not id:
            id = f"id='{id}'"

    def __serialize__(self) -> Any:
        return serialize(self.get(Entry.op_tag.dump))

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
        obj.put(Entry.op_tag.assign, desc)
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

    def __post_process__(self, value: _T,   *args,   query=_undefined_,  **kwargs) -> Union[_T, _TNode]:

        if isinstance(value, (int, float, str, np.ndarray, Node)) or value in (None, _not_found_, _undefined_):
            return value
        elif isinstance(value, collections.abc.Sequence):
            value = List(value, *args, parent=self, **kwargs)
        elif isinstance(value, collections.abc.Mapping):
            value = Dict(value, *args, parent=self, **kwargs)
        elif isinstance(value, Entry):
            value = Node(value, *args, parent=self, **kwargs)

        return value

    def get(self, query: _TQuery = None,  default_value: _T = _undefined_,  **kwargs) -> _T:
        return super().get(query, default_value, **kwargs)

    def put(self, query: _TQuery, value: _T, /, **kwargs) -> Tuple[_T, bool]:
        return super().put(query,  value)

    def fetch(self, query: _TQuery = None,  default_value: _T = _undefined_,  **kwargs) -> _T:
        return self.__post_process__(super().get(query, default_value, **kwargs), query=query)

    def remove(self, query: _TQuery, /, **kwargs) -> None:
        return super().remove(query,  **kwargs)

    def __setitem__(self, query: _TQuery, value: _T) -> _T:
        return self.put(query,  self.__pre_process__(value))

    def __getitem__(self, query: _TQuery) -> _TNode:
        return self.__post_process__(self.get(query), query=query)

    def __delitem__(self, query: _TQuery) -> bool:
        return self.put(query, Entry.op_tag.erase)

    def __contains__(self, query: _TQuery) -> bool:
        return self.get(query, Entry.op_tag.exists)

    def __len__(self) -> int:
        return self._entry.pull(Entry.op_tag.count)

    def __iter__(self) -> Iterator[_T]:
        for idx, obj in enumerate(self._entry.iter()):
            yield self.__post_process__(obj, query=[idx])

    def __eq__(self, other) -> bool:
        return self._entry.pull({Entry.op_tag.equal: other})

    def __bool__(self) -> bool:
        return not self.empty  # and (not self.__fetch__())

    def _as_dict(self) -> Mapping:
        return {k: self.__post_process__(v, query=[k]) for k, v in self._entry.items()}

    def _as_list(self) -> Sequence:
        return [self.__post_process__(v, query=[idx]) for idx, v in enumerate(self._entry.values())]

    def _as_type(self, prop_type, value, parent=None, **kwargs):
        if (inspect.isclass(prop_type) and issubclass(prop_type, Node)) or issubclass(getattr(prop_type, '__origin__', type(None)), Node):
            return prop_type(value, parent=parent, **kwargs)
        elif dataclasses.is_dataclass(prop_type):
            if isinstance(value, collections.abc.Mapping):
                return prop_type(**{k: value.get(k, None) for k in prop_type.__dataclass_fields__})
            elif isinstance(value, prop_type):
                return value
            else:
                raise TypeError(type(value))
        elif callable(prop_type):
            return prop_type(value,  **kwargs)
        else:
            return value

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
    __slots__ = ("_v_kwargs", "__new_child__")

    def __init__(self, cache: Optional[Sequence] = None, /,  parent=None, new_child=_undefined_, **kwargs) -> None:
        if not isinstance(cache, (collections.abc.Sequence, Entry)) or isinstance(cache, str):
            cache = [cache]
        Node.__init__(self, cache if cache is not None else _LIST_TYPE_(),  parent=parent)
        self._v_kwargs = kwargs
        self.__new_child__ = new_child

    def __serialize__(self) -> Sequence:
        return [serialize(v) for v in self._as_list()]

    def __post_process__(self, value: _T,  /,  query=_undefined_,  **kwargs) -> Union[_T, _TNode]:
        if isinstance(value, (int, float, str, np.ndarray, Node)) \
                or value in (None, _not_found_, _undefined_):
            return value
        elif (isinstance(query, list) and (len(query) == 0 or isinstance(query[-1], str))):
            return value
        elif not isinstance(query, list):
            parent = self
            key = query
        elif len(query) == 1:
            parent = self
            key = query[0]
        else:
            parent = super().get(query[:-1])
            key = query[-1]

        if self.__new_child__ is _undefined_:
            child_cls = None
            #  @ref: https://stackoverflow.com/questions/48572831/how-to-access-the-type-arguments-of-typing-generic?noredirect=1
            orig_class = getattr(self, "__orig_class__", None)
            if orig_class is not None:
                child_cls = get_args(self.__orig_class__)
                if child_cls is not None and len(child_cls) > 0 and inspect.isclass(child_cls[0]):
                    child_cls = child_cls[0]
            self.__new_child__ = child_cls

        n_value = self._as_type(self.__new_child__, value, **collections.ChainMap(kwargs, self._v_kwargs))
        if n_value is not value and key is not _undefined_:
            parent.put(key, n_value)
        return n_value

    @property
    def __category__(self):
        return super().__category__ | Node.Category.LIST

    def __len__(self) -> int:
        return super().__len__()

    def __setitem__(self, query: _TQuery, v: _T) -> None:
        super().__setitem__(query,  v)

    def __getitem__(self, query: _TQuery) -> _T:
        return super().__getitem__(query)

    def __delitem__(self, query: _TQuery) -> None:
        return super().__delitem__(query)

    def __iter__(self) -> Iterator[_T]:
        for idx, obj in enumerate(self._entry.iter()):
            yield self.__post_process__(obj, query=[idx])

    def __iadd__(self, other):
        self.put(_next_, other)
        return self

    def sort(self):
        if hasattr(self._entry.__class__, "sort"):
            self._entry.sort()
        else:
            raise NotImplementedError()

    def combine(self, default_value=None, reducer=_undefined_, partition=_undefined_) -> _T:
        return self.__post_process__(EntryCombiner(self, default_value=default_value,  reducer=reducer, partition=partition),  parent=self._parent)

    def refresh(self, d=None, /, **kwargs):
        # super().update(d)
        for element in self.__iter__():
            if hasattr(element.__class__, 'refresh'):
                element.refresh(**kwargs)

    def reset(self, value=None):
        if isinstance(value, (collections.abc.Sequence)):
            super().reset(value)
        else:
            self._combine = value
            super().reset()


class Dict(Node[_T], Mapping[str, _T]):
    __slots__ = ("__new_child__")

    def __init__(self, cache: Optional[Mapping] = None,  /, new_child: Callable = None, **kwargs):
        Node.__init__(self, cache if cache is not None else _DICT_TYPE_(),   **kwargs)
        self.__new_child__ = new_child

    def __post_process__(self, value: _T,   *args, parent=None, query=_undefined_,  **kwargs) -> Union[_T, _TNode]:
        if isinstance(value, (int, float, str, np.ndarray, Node)) \
                or value in (None, _not_found_, _undefined_) \
                or not isinstance(query, (list, str)) \
                or len(query) == 0 \
                or not isinstance(query[-1], str):
            return value
        elif not isinstance(query, list):
            parent = self
            key = query
        else:
            parent = super().get(query[:-1])
            key = query[-1]

        if self.__new_child__ is not None:
            n_value = self._as_type(self.__new_child__, value, **kwargs)
        else:
            # FIXME: Needs optimization
            prop = dict(inspect.getmembers(parent.__class__)).get(key, _not_found_)

            prop_type = _undefined_
            if isinstance(prop, (_SpProperty, cached_property)):
                prop_type = prop.func.__annotations__.get("return", None)
            elif isinstance(prop, (property)):
                prop_type = prop.fget.__annotations__.get("return", None)

            n_value = self._as_type(prop_type, value, **kwargs)

        if n_value is not value:
            parent.put(key, n_value)

        return n_value

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
        yield from super().__iter__()

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o)

    def __contains__(self, o: object) -> bool:
        return super().__contains__(o)

    def __ior__(self, other):
        return self.put(Entry.op_tag.update, other)

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

    def __get__(self, instance: Node, owner=None) -> _T:
        if not isinstance(instance, Node):
            cache = Entry(instance.__dict__)
        else:
            cache = instance

        if self.attrname is None:
            raise TypeError("Cannot use _SpProperty instance without calling __set_name__ on it.")

        val = cache.get(self.attrname, default_value=_not_found_)

        if not self._isinstance(val):
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = cache.get(self.attrname, default_value=_not_found_)
                # FIXME: Thread safety is not guaranteed! solution: lock on cache???
                if not self._isinstance(val):
                    obj = self.func(instance)
                    if not self._isinstance(obj):
                        val = instance.__post_process__(obj, query=self.attrname)
                    elif obj is not _undefined_ and obj is not val and isinstance(cache, Entry):
                        val = cache.put(self.attrname, obj)
                    else:
                        val = obj
                    # if not self._isinstance(obj) and getattr(instance, '__new_child__', None) not in (None, _not_found_, _undefined_):
                    #     obj = instance.__new_child__(obj)
                    # if not self._isinstance(obj):
                    #     origin_type = getattr(self.return_type, '__origin__', self.return_type)
                    #     if dataclasses.is_dataclass(origin_type):
                    #         obj = as_dataclass(origin_type, obj)
                    #     elif inspect.isclass(origin_type) and issubclass(origin_type, Node):
                    #         obj = self.return_type(obj, parent=instance)
                    #     elif origin_type is np.ndarray:
                    #         obj = np.asarray(obj)
                    #     elif callable(self.return_type) is not None:
                    #         try:
                    #             tmp = self.return_type(obj)
                    #         except Exception as error:
                    #             logger.error(f"{self.attrname} {self.return_type} {type(obj)} : {error}")
                    #             raise error
                    #         else:
                    #             obj = tmp

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


def chain_map(*args, **kwargs) -> collections.ChainMap:
    d = []
    for a in args:
        if isinstance(d, collections.abc.Mapping):
            d.append(a)
        elif isinstance(d, Entry):
            d.append(Dict(a))
    return collections.ChainMap(*d, **kwargs)
