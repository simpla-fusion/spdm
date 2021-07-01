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

from numpy.lib.arraysetops import _isin_dispatcher, isin

from ..numlib import np, scipy
from ..util.logger import logger
from ..util.sp_export import sp_find_module
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

    __slots__ = "__orig_class__", "_parent",  "_new_child"

    def __init__(self, entry: Any = None, /, parent=None, **kwargs):
        super().__init__(entry, **kwargs)
        self._parent = parent

    def __repr__(self) -> str:
        annotation = [f"{k}='{v}'" for k, v in self.annotation.items() if v is not None]
        return f"<{getattr(self,'__orig_class__',self.__class__.__name__)} {' '.join(annotation)}/>"

    @property
    def annotation(self) -> dict:
        return {
            "id": self.get("@id", None),
            "type":  self._entry.__class__.__name__
        }

    @property
    def nid(self) -> str:
        return self.get("@id", None)

    def _property_type(self, property_name=_undefined_) -> _T:
        prop_type = _undefined_

        if property_name is not _undefined_:
            prop = dict(inspect.getmembers(self.__class__)).get(property_name, _not_found_)

            if isinstance(prop, (_SpProperty, cached_property)):
                prop_type = prop.func.__annotations__.get("return", None)
            elif isinstance(prop, (property)):
                prop_type = prop.fget.__annotations__.get("return", None)

        elif self._new_child is _undefined_:
            child_cls = Node
            #  @ref: https://stackoverflow.com/questions/48572831/how-to-access-the-type-arguments-of-typing-generic?noredirect=1
            orig_class = getattr(self, "__orig_class__", None)
            if orig_class is not None:
                child_cls = get_args(self.__orig_class__)
                if child_cls is not None and len(child_cls) > 0 and inspect.isclass(child_cls[0]):
                    child_cls = child_cls[0]
            self._new_child = child_cls
            prop_type = self._new_child
        else:
            prop_type = self._new_child

        return prop_type

    def _convert(self, value: _T,   *args,  property_type=_undefined_, property_name=_undefined_,  **kwargs) -> Union[_T, _TObject]:
        if property_type is not _undefined_:
            pass
        elif isinstance(value, (int, float, str, np.ndarray, Node)) or value in (None, _not_found_, _undefined_):
            return value
        elif isinstance(value, collections.abc.Sequence):
            return List(value, *args, parent=self, **kwargs)
        elif isinstance(value, collections.abc.Mapping):
            return Dict(value, *args, parent=self, **kwargs)
        elif isinstance(value, Entry):
            return Node(value, *args, parent=self, **kwargs)

        elif prop_type in (int, float):
            return prop_type(value)
        elif prop_type is np.ndarray:
            return np.asarray(value)
        elif (inspect.isclass(prop_type) and issubclass(prop_type, Node)) or issubclass(getattr(prop_type, '__origin__', type(None)), Node):
            return prop_type(value, parent=parent, **kwargs)
        elif dataclasses.is_dataclass(prop_type):
            if isinstance(value, collections.abc.Mapping):
                return prop_type(**{k: value.get(k, None) for k in prop_type.__dataclass_fields__})
            elif isinstance(value, prop_type):
                return value
            else:
                raise TypeError(type(value))
        elif callable(prop_type):
            return prop_type(value, **kwargs)
        else:
            return value

    def _serialize(self) -> Any:
        return serialize(self.get(Entry.op_tag.dump))

    @classmethod
    def _deserialize(cls, desc: Any) -> _TNode:
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

    def _duplicate(self) -> _TNode:
        obj = super()._duplicate()
        obj._parent = self._parent
        return obj

    def __hash__(self) -> int:
        return NotImplemented

    def _pre_process(self, value: Any, *args, **kwargs) -> Any:
        return value

    def _post_process(self, value: _T,   *args,     **kwargs) -> Union[_T, _TNode]:
        return self._convert(value, *args, **kwargs)

    def get(self, query: _TQuery = None,  default_value: _T = _undefined_,  **kwargs) -> _T:
        return super().get(query, default_value, **kwargs)

    def put(self, query: _TQuery, value: _T, /, **kwargs) -> Tuple[_T, bool]:
        return super().put(query,  value)

    def fetch(self, query: _TQuery = None,  default_value: _T = _undefined_,  **kwargs) -> _T:
        return self._post_process(super().get(query, default_value, **kwargs))

    def remove(self, query: _TQuery, /, **kwargs) -> None:
        return super().remove(query,  **kwargs)

    def __setitem__(self, query: _TQuery, value: _T) -> _T:
        return self.put(query,  self._pre_process(value))

    def __getitem__(self, query: _TQuery) -> _TNode:
        return self._post_process(self.get(query))

    def __delitem__(self, query: _TQuery) -> bool:
        return self.put(query, Entry.op_tag.erase)

    def __contains__(self, query: _TQuery) -> bool:
        return self.get(query, Entry.op_tag.exists)

    def __len__(self) -> int:
        return self._entry.pull(Entry.op_tag.count)

    def __iter__(self) -> Iterator[_T]:
        for obj in self._entry.iter():
            yield self._post_process(obj)

    def __eq__(self, other) -> bool:
        return self._entry.pull({Entry.op_tag.equal: other})

    def __bool__(self) -> bool:
        return not self.empty  # and (not self.__fetch__())

    def _as_dict(self) -> Mapping:
        return {k: self._post_process(v) for k, v in self._entry.items()}

    def _as_list(self) -> Sequence:
        return [self._post_process(v) for v in self._entry.values()]

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
    __slots__ = ()

    def __init__(self, cache: Union[Sequence, Entry] = None, /,   **kwargs) -> None:
        if cache is None:
            cache = _LIST_TYPE_()
        elif isinstance(cache, Entry):
            cache = cache.pull(_LIST_TYPE_())

        Node.__init__(self, cache, **kwargs)

        if isinstance(cache, collections.abc.Sequence):
            cache = [self._convert(v) for v in cache]

    def _serialize(self) -> Sequence:
        return [serialize(v) for v in self._as_list()]

    @classmethod
    def _deserialize(cls, desc: Any) -> _TNode:
        return NotImplemented

    def __len__(self) -> int:
        return super().__len__()

    def __setitem__(self, query: _TQuery, v: _T) -> None:
        super().__setitem__(query,  v)

    def __getitem__(self, query: _TQuery) -> _T:
        return super().__getitem__(query)

    def __delitem__(self, query: _TQuery) -> None:
        return super().__delitem__(query)

    def __iter__(self) -> Iterator[_T]:
        for obj in self._entry.iter():
            yield self._post_process(obj)

    def __iadd__(self, other):
        self.put(_next_, other)
        return self

    def sort(self):
        if hasattr(self._entry.__class__, "sort"):
            self._entry.sort()
        else:
            raise NotImplementedError()

    def combine(self, default_value=None, reducer=_undefined_, partition=_undefined_) -> _T:
        return self._post_process(EntryCombiner(self, default_value=default_value,  reducer=reducer, partition=partition))

    def refresh(self, *args, **kwargs):
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

    def find(self, condition,  only_first=True) -> _T:
        return self._post_process(self._entry.pull(condition=condition, only_first=only_first))

    def update(self, d, predication=_undefined_, only_first=False) -> int:
        return self._entry.push(self._pre_process(d), predication=predication, only_first=only_first)


class Dict(Node[_T], Mapping[str, _T]):
    __slots__ = ()

    def __init__(self, cache: Optional[Mapping] = None,  /,  **kwargs):
        if cache is None:
            cache = _DICT_TYPE_()
        elif isinstance(cache, Entry):
            cache = cache.pull(_DICT_TYPE_())

        Node.__init__(self, cache,   **kwargs)

    def _serialize(self) -> Mapping:
        return {k: serialize(v) for k, v in self._as_dict()}

    @classmethod
    def _deserialize(cls, desc: Any) -> _TNode:
        return NotImplemented

    def __getitem__(self, key: str) -> _T:
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: _T) -> None:
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)

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
    #             # elif hasattr(v, "_serialize"):
    #             #     res[k] = v._serialize()
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
    def __init__(self, func: Callable[..., _T]):
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

    def __put__(self, cache: Any, val: Any):
        if isinstance(val, Node):
            logger.debug((self.attrname, type(val._entry), type(cache), val._entry._data is cache._data))

        try:
            cache.insert(self.attrname, val)
        except TypeError as error:
            # logger.error(f"Can not put value to '{self.attrname}'")
            raise TypeError(error) from None

    def __set__(self, instance: Any, value: Any):
        with self.lock:
            cache = getattr(instance, "_entry", Entry(instance.__dict__))
            cache.insert(self.attrname, value, assign_if_exists=True)

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def __get__(self, instance: Node, owner=None) -> _T:
        if instance is None:
            return self
        try:
            cache = instance._entry
        except AttributeError as error:
            logger.exception(error)
            raise AttributeError(error)

        if not isinstance(instance, Node):
            raise TypeError(f"{type(instance)} {self.attrname}")

        if self.attrname is None:
            raise TypeError("Cannot use sp_property instance without calling __set_name__ on it.")

        val = cache.get(self.attrname, _not_found_)
        if val is _not_found_ or not self._isinstance(val):
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = cache.get(self.attrname, _not_found_)
                if val is _not_found_ or not self._isinstance(val):
                    val = self.func(instance)

                    if not self._isinstance(val):
                        val = instance._post_process(val, query=self.attrname)

                    cache.put(self.attrname, val)

        return val


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
