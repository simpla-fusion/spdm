import bisect
import collections
import collections.abc
import dataclasses
import enum
import inspect
import pprint
from _thread import RLock
from enum import IntFlag
from functools import cached_property
from typing import (Any, Callable, Generic, Iterator, Mapping, MutableMapping,
                    MutableSequence, Optional, Sequence, Tuple, Type, TypeVar,
                    Union, final, get_args)

from ..numlib import np, scipy
from ..util.logger import logger
from ..util.utilities import _not_found_, _undefined_, serialize
from .Entry import (_DICT_TYPE_, _LIST_TYPE_, Entry, EntryCombiner,
                    EntryWrapper, _next_, _TIndex, _TKey, _TObject, _TPath,
                    _TQuery, as_dataclass, normalize_query)


class Node(Generic[_TObject]):
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
    __slots__ = "_parent", "_entry",  "__orig_class__", "__new_child__"

    def __init__(self, cache: Any = None, *args, parent=None, new_child=_undefined_, writable=True, **kwargs):
        super().__init__()
        self._parent = parent
        self.__new_child__ = new_child

        if isinstance(cache, Node):
            self._entry = cache._entry
        elif not isinstance(cache, Entry):
            self._entry = Entry(cache)
        elif isinstance(cache, EntryCombiner):
            self._entry = cache
        elif not cache.writable:
            self._entry = EntryWrapper(Entry(), cache)
        else:
            self._entry = cache

    def __repr__(self) -> str:
        return f"<{getattr(self,'__orig_class__',self.__class__.__name__)} />"
        # return pprint.pformat(self.__serialize__())

    def __serialize__(self) -> Mapping:
        return serialize(self._entry.find(full=True))

    def __duplicate__(self, desc=None) -> object:
        return self.__class__(collections.ChainMap(desc or {}, self.__serialize__()), parent=self._parent)

    def _as_dict(self) -> Mapping:
        return {k: self.__post_process__(v) for k, v in self._entry.items()}

    def _as_list(self) -> Sequence:
        return [self.__post_process__(v) for v in self._entry.values()]

    @property
    def __parent__(self) -> object:
        return self._parent

    def __hash__(self) -> int:
        return NotImplemented

    def __clear__(self) -> None:
        self._entry.clear()

    @property
    def empty(self) -> bool:
        return self._entry.empty or len(self) == 0

    @property
    def invalid(self) -> bool:
        return self._entry is None or self._entry.invalid

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

    def __post_process__(self, d: Any, key=None, /, *args, parent=None, **kwargs) -> _TObject:
        if d is _not_found_:
            return d

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
        value = d

        if inspect.isclass(self.__new_child__):
            if isinstance(value, self.__new_child__):
                pass
            elif isinstance(value, collections.abc.Sequence) and not isinstance(value, str) and not issubclass(self.__new_child__, List):
                value = List[self.__new_child__](value, *args, parent=parent, **kwargs)
            elif isinstance(value, collections.abc.Mapping) and not issubclass(self.__new_child__, Dict):
                value = Dict[self.__new_child__](value, *args, parent=parent, **kwargs)
            elif issubclass(self.__new_child__, Node):
                value = self.__new_child__(value, *args, parent=parent, **kwargs)
            else:
                logger.warning(f"{self.__new_child__} is not a 'Node'!")
                value = self.__new_child__(value, *args, **kwargs)
        elif callable(self.__new_child__):
            value = self.__new_child__(value, *args, parent=parent,  **kwargs)
        elif self.__new_child__ is not None:
            raise TypeError(f"Illegal type! {self.__new_child__}")

        if isinstance(value, Node):
            pass
        elif isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
            value = List(value, *args, parent=parent, new_child=self.__new_child__,  **kwargs)
        elif isinstance(value, collections.abc.Mapping):
            value = Dict(value, *args, parent=parent, new_child=self.__new_child__,  **kwargs)
        elif isinstance(value, Entry):
            value = Node(value, *args, parent=parent, new_child=self.__new_child__, **kwargs)

        if isinstance(key, (int, str)) and d is not value and self._entry.writable:
            try:
                self.__setitem__(key, value)
            except Exception:
                pass
        return value

    def __setitem__(self, query: _TQuery, value: Any) -> None:
        self._entry.insert(query,  self.__pre_process__(value), assign_if_exists=True)

    def __getitem__(self, query: _TQuery) -> Any:
        return self.__post_process__(self._entry.find(query))

    def __delitem__(self, query: _TQuery) -> None:
        self._entry.erase(query)

    def __contains__(self, query: _TQuery) -> bool:
        return self._entry.contains(query)

    def __len__(self) -> int:
        return self._entry.count()

    def __iter__(self) -> Iterator[_TObject]:
        for obj in self._entry.iter():
            yield self.__post_process__(obj)

    def __eq__(self, other) -> bool:
        return self._entry.compare(other)

    # def __fetch__(self):
    #     if hasattr(self._entry.__class__, "fecth"):
    #         self._entry = self._entry.fetch()
    #     return self._entry

    def __bool__(self) -> bool:
        return not self.empty and (not self.__fetch__())

    # def __array__(self) -> np.ndarray:
    #     return np.asarray(self.__fetch__())

    def find(self, *query,  **kwargs) -> _TObject:
        return self._entry.find(list(query),  **kwargs)

    def insert(self, query: _TQuery, value: Any, /,  **kwargs) -> _TObject:
        return self._entry.insert(query, value, **kwargs)

    def insert_or_assign(self, query: _TQuery, value: Any, /,  **kwargs) -> _TObject:
        return self._entry.insert(query, value, assign_if_exists=True, **kwargs)

    def get(self, query: _TQuery = None,  default_value=_undefined_):
        return self.find(query, only_first=True, default_value=default_value)

    def update(self, *args, **kwargs):
        self._entry.update(*args, **kwargs)

    def update_many(self,  *args,  **kwargs):
        self._entry.update_many(*args,  **kwargs)

    def fetch(self, query, /, default_value=None, **kwargs):
        return self._entry.find(query, default_value=default_value, **kwargs)


class List(Node[_TObject], Sequence[_TObject]):
    __slots__ = ("_v_args", )

    def __init__(self, cache: Optional[Sequence] = None, *args, parent=None,   **kwargs) -> None:
        if cache is None or cache is _not_found_:
            cache = _LIST_TYPE_()
        Node.__init__(self, cache, *args, parent=parent, **kwargs)
        self._v_args = (args, kwargs)

    def __serialize__(self) -> Sequence:
        return [serialize(v) for v in self._as_list()]

    @property
    def __category__(self):
        return super().__category__ | Node.Category.LIST

    def __len__(self) -> int:
        return self._entry.count()

    def __setitem__(self, query: _TQuery, v: _TObject) -> None:
        self._entry.insert(query, self.__pre_process__(v), assign_if_exists=True)

    def __getitem__(self, query: _TQuery) -> _TObject:
        return self.__post_process__(self._entry.find(query, only_first=True), query, parent=self._parent)

    def __delitem__(self, query: _TQuery) -> None:
        super().__delitem__(query)

    def __iter__(self) -> Iterator[_TObject]:
        for idx in range(len(self)):
            yield self.__getitem__(idx)

    def __iadd__(self, other):
        self._entry.insert(_next_, self.__pre_process__(other))
        return self

    def sort(self):
        if hasattr(self._entry.__class__, "sort"):
            self._entry.sort()
        else:
            raise NotImplementedError()

    @property
    def combine(self) -> _TObject:
        return self.__new_child__(EntryCombiner(self._entry), parent=self._parent)

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        for element in self.__iter__():
            if hasattr(element.__class__, 'update'):
                element.update(*args, **kwargs)


class Dict(Node[_TObject], Mapping[str, _TObject]):
    __slots__ = ()

    def __init__(self, cache: Optional[Mapping] = None, *args,  **kwargs):

        if cache is None or cache is _not_found_:
            cache = _DICT_TYPE_()
        elif isinstance(cache, Node):
            cache = cache._entry

        Node.__init__(self, cache, *args, **kwargs)

    def __serialize__(self) -> Mapping:
        return {k: serialize(v) for k, v in self._as_dict().items()}

    @classmethod
    def __deserialize__(cls, desc: Mapping) -> _TObject:
        ids = desc.get("@ids", None)
        if ids is None:
            raise ValueError(desc)
        else:
            raise NotImplementedError(ids)

    @property
    def __category__(self):
        return super().__category__ | Node.Category.LIST

    def __getitem__(self, query: _TKey) -> _TObject:
        return self.__post_process__(self._entry.find(query), query)

    def __setitem__(self, query: _TKey, value: _TObject) -> None:
        self._entry.insert(query,  self.__pre_process__(value), assign_if_exists=True)

    def __delitem__(self, query: _TKey) -> None:
        self._entry.erase(query)

    def __iter__(self) -> Iterator[str]:
        yield from self.keys()

    def __len__(self) -> int:
        return self._entry.count()

    def __eq__(self, o: object) -> bool:
        return self._entry.compare(o)

    def __contains__(self, o: object) -> bool:
        return self._entry.contains(o)

    def __ior__(self, other):
        return self.update(other)

    def update(self, value: Any = None, *args, **kwargs) -> None:
        self._entry.update(None, value, *args, **kwargs)

    def get(self, key: _TQuery, default_value=_not_found_, **kwargs) -> _TObject:
        return self._entry.find(key, default_value=default_value, **kwargs)

    def items(self) -> Iterator[Tuple[str, _TObject]]:
        for k, v in self._entry.items():
            yield k, self.__post_process__(v, k)

    def keys(self) -> Iterator[str]:
        yield from self._entry.keys()

    def values(self) -> Iterator[_TObject]:
        for k, v in self._entry.items():
            yield self.__post_process__(v, k)

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


class _SpProperty(Generic[_TObject]):
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

    def __get__(self, instance: Any, owner=None) -> _TObject:
        cache = getattr(instance, "_entry", Entry(instance.__dict__))

        if self.attrname is None:
            raise TypeError("Cannot use _SpProperty instance without calling __set_name__ on it.")
        # elif isinstance(cache, Entry) and not cache.writable:
        #     logger.error(f"Attribute cache is not writable!")
        #     raise AttributeError(self.attrname)

        val = cache.find(self.attrname, default_value=_not_found_)

        if not self._isinstance(val):
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = cache.find(self.attrname, default_value=_not_found_)
                # FIXME: Thread safety cannot be guaranteed! solution: lock on cache
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
                                logger.error(f"{self.attrname} {self.return_type} {type(val)} : {error}")
                                raise error
                            else:
                                obj = tmp

                    if obj is not val and not getattr(obj, 'invalid', False) and isinstance(cache, Entry) and cache.writable:
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


def sp_property(func: Callable[..., _TObject]) -> _SpProperty[_TObject]:
    return _SpProperty[_TObject](func)
