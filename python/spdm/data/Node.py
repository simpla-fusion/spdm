import bisect
import collections
import collections.abc
import copy
import enum
import functools
import inspect
import pprint
import typing
from enum import IntFlag
from functools import cached_property
from typing import (Any, Generic, Iterable, Iterator, Mapping, MutableMapping,
                    MutableSequence, Optional, Sequence, TypeVar, Union,
                    get_args)

import numpy as np
from matplotlib.pyplot import isinteractive
from numpy.lib.function_base import iterable
from sympy.core import cache

from ..util.logger import logger
from ..util.utilities import _not_defined_, _not_found_, serialize
from .Entry import Entry, _last_, _next_, _TIndex, _TKey, _TPath

_TObject = TypeVar('_TObject')


class Node(object):
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
    __slots__ = "_parent", "_entry", "_default_factory", "__orig_class__"

    def __init__(self, data: Any = None, *args, default_factory=None, parent=None, **kwargs):
        super().__init__()
        self._default_factory = default_factory
        self._parent = parent

        if isinstance(data, Node):
            self._entry = data._entry
        elif isinstance(data, Entry):
            self._entry = data
        else:
            self._entry = Entry(data, *args, **kwargs)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} />"
        # return pprint.pformat(self.__serialize__())

    def __serialize__(self) -> Mapping:
        if isinstance(self._entry, Entry):
            return f"<{self.__class__.__name__} type={self._entry.__class__.__name__} path={ self._entry._prefix}>"
        else:
            return serialize(self._entry)

    def __duplicate__(self, desc=None) -> object:
        return self.__class__(desc if desc is not None else self.__serialize__(), parent=self._parent)

    def _as_dict(self) -> Mapping:
        return Entry._DICT_TYPE_

    def _as_list(self) -> Sequence:
        return Entry._LIST_TYPE_

    @property
    def __parent__(self) -> object:
        return self._parent

    def __hash__(self) -> int:
        return hash(self._name)

    def __clear__(self) -> None:
        self._entry = None

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
        if isinstance(d, (Entry)):
            flag |= Node.Category.ENTRY
        elif isinstance(d, np.ndarray):
            flag |= Node.Category.ARRAY
            if np.issubdtype(d.dtype, np.int64):
                flag |= Node.Category.INT
            elif np.issubdtype(d.dtype, np.float64):
                flag |= Node.Category.FLOAT
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

        return flag

    def empty(self) -> bool:
        return self._entry is None or (isinstance(self._entry, (collections.abc.Sequence, collections.abc.Mapping)) and len(self._entry) == 0)

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
    @property
    def __category__(self) -> Category:
        return Node.__type_category__(self._entry)

    def __genreric_template_arguments__(self):
        #  @ref: https://stackoverflow.com/questions/48572831/how-to-access-the-type-arguments-of-typing-generic?noredirect=1
        orig_class = getattr(self, "__orig_class__", None)
        if orig_class is not None:
            return get_args(self.__orig_class__)
        else:
            return None

    def __check_template__(self, cls) -> bool:
        orig_class = getattr(self, "__orig_class__", None)
        if orig_class is not None:
            return issubclass(cls,  get_args(orig_class))
        else:
            return issubclass(cls, Node)

    @property
    def __factory__(self):
        if self._default_factory is None:
            factory = self.__genreric_template_arguments__()
            if factory is not None and len(factory) > 0 and inspect.isclass(factory[0]):
                self._default_factory = factory[0]
        return self._default_factory

    def __new_child__(self, value, *args, parent=None,  **kwargs) -> Any:
        parent = parent if parent is not None else self
        if isinstance(value, Node):
            pass
        elif self.__factory__ is not None:
            value = self.__factory__(value, *args,  parent=parent, ** kwargs)
        elif isinstance(value, collections.abc.MutableSequence):
            value = List(value, *args, parent=parent, **kwargs)
        elif isinstance(value, collections.abc.MutableMapping):
            value = Dict(value, *args, parent=parent,  **kwargs)
        # elif isinstance(value, Entry):
        #     value = Node(value, *args, parent=parent, *kwargs)

        return value

    def __pre_process__(self, value: Any, *args, **kwargs) -> Any:
        return value

    def __post_process__(self, value: Any,  *args,   **kwargs) -> Any:
        if self._entry.writable and not isinstance(value, Entry):
            self._entry.put(value, *args, **kwargs)
        return value

    def __fetch__(self, path: Optional[_TPath] = None, default_value=None) -> Any:
        obj = self._entry.get(path)
        if isinstance(obj, Entry):
            obj = default_value
        return obj

    def __setitem__(self, path: _TPath, value: Any) -> None:
        self._entry.put(self.__pre_process__(value), path)

    def __getitem__(self, path: _TPath) -> Any:
        return self.__post_process__(self._entry.get(path), path)

    def __delitem__(self, path: _TPath) -> None:
        if isinstance(self._entry, Entry):
            self._entry.delete(path)
        elif not path:
            self.__clear__()
        elif not isinstance(self._entry, (collections.abc.Mapping, collections.abc.MutableSequence)):
            raise TypeError(type(self._entry))
        else:
            del self._entry[path]

    def __contains__(self, path: _TPath) -> bool:
        if isinstance(self._entry, Entry):
            return self._entry.contains(path, None)
        elif isinstance(path, str) and isinstance(self._entry, collections.abc.Mapping):
            return path in self._entry
        elif not isinstance(path, str):
            return path >= 0 and path < len(self._entry)
        else:
            return False

    def __len__(self) -> int:
        if isinstance(self._entry, Entry):
            return self._entry.count()
        elif isinstance(self._entry,  (collections.abc.Mapping, collections.abc.MutableSequence)):
            return len(self._entry)
        else:
            return 0 if self._entry is _not_found_ else 1

    def __iter__(self) -> Iterator[_TObject]:
        if isinstance(self._entry, Entry):
            for idx, obj in enumerate(self._entry.iter()):
                yield self.__post_process__(obj, idx)
        else:
            d = self._entry.get()

            if isinstance(d, Entry):
                yield from d.iter()
            elif isinstance(d, (collections.abc.MutableSequence)):
                # yield from map(lambda idx, v: self.__post_process__(v, idx), enumerate(d))
                for idx, v in enumerate(d):
                    yield self.__post_process__(v, idx)
            elif isinstance(d, collections.abc.Mapping):
                # yield from map(lambda idx, v: self.__post_process__(v, idx), d.items())
                for idx, v in d.items():
                    yield self.__post_process__(v, idx)
            else:
                yield self.__post_process__(d)

    def __eq__(self, other) -> bool:
        if isinstance(other, Node):
            other = other._entry

        if isinstance(self._entry, Entry):
            return other is None or self._entry.equal(other)
        else:
            return self._entry == other

    def __bool__(self) -> bool:
        return False if isinstance(self._entry, Entry) else (not not self._entry)


class List(Node, MutableSequence[_TObject]):
    __slots__ = ()

    def __init__(self, data: Optional[Sequence] = None, *args,  **kwargs) -> None:
        if data == None and data is _not_found_:
            data = []
        Node.__init__(self, data, *args, **kwargs)

    def __serialize__(self) -> Sequence:
        return [serialize(v) for v in self._as_list()]

    def _as_list(self) -> Sequence:
        return [v for v in self._entry.values()]

    @property
    def __category__(self):
        return super().__category__() | Node.Category.LIST

    def __len__(self) -> int:
        return Node.__len__(self)

    def __new_child__(self, value, *args, parent=None,  **kwargs) -> Any:
        return super().__new_child__(value, *args, parent=parent if parent is not None else self._parent, **kwargs)

    def __post_process__(self, value: Any, *args, **kwargs) -> Any:
        return super().__post_process__(self.__new_child__(value), *args, **kwargs)

    def __setitem__(self, k: _TIndex, v: _TObject) -> None:
        Node.__setitem__(self, k, v)

    def __getitem__(self, k: _TIndex) -> _TObject:
        return Node.__getitem__(self, k)

    def __delitem__(self, k: _TIndex) -> None:
        Node.__delitem__(self, k)

    def __iter__(self) -> Iterator[_TObject]:
        yield from Node.__iter__(self)

    def insert(self, idx, value=None, sorted=True) -> _TObject:
        if value is None:
            value = idx
            idx = None
        if value is None:
            pass
        elif not self.__check_template__(value.__class__):
            value = self.__new_child__(value)

        if idx is not None:
            self._entry.put(value, idx)
        elif not sorted:
            self._entry.put(value, -1)
        elif isinstance(self._entry, Entry):
            data = self._entry._data
            if not isinstance(data, collections.abc.MutableSequence):
                raise NotImplementedError(f"{type(data)} is not  MutableSequence!")
            else:
                idx = bisect.bisect_right(data, value)
                data.insert(idx, value)
        else:
            raise TypeError(type(self._entry))
        return value

    def find_first(self, func):
        idx, v = next(filter(lambda t: func(t[1]), enumerate(self._entry)))
        return idx, v

    def sort(self):
        if hasattr(self._entry.__class__, "sort"):
            self._entry.sort()
        else:
            raise NotImplementedError()


class Dict(Node, MutableMapping[_TKey, _TObject]):
    __slots__ = ()

    def __init__(self, data: Optional[Mapping] = None, *args,  **kwargs):
        if data is None or data is _not_found_:
            data = {}
        Node.__init__(self, data, *args, **kwargs)

    def __serialize__(self, properties: Optional[Sequence] = None) -> Mapping:
        return {k: serialize(v) for k, v in self._as_dict().items() if properties is None or k in properties}

    @classmethod
    def __deserialize__(cls, desc: Mapping) -> _TObject:
        ids = desc.get("@ids", None)
        if ids is None:
            raise ValueError(desc)
        else:
            raise NotImplementedError(ids)

    def _as_dict(self) -> Mapping:
        cls = self.__class__
        if cls is Dict:
            return self._entry._data
        else:
            properties = set([k for k in self.__dir__() if not k.startswith('_')])
            res = {}
            for k in properties:
                prop = getattr(cls, k, None)
                if inspect.isfunction(prop) or inspect.isclass(prop) or inspect.ismethod(prop):
                    continue
                elif isinstance(prop, cached_property):
                    v = prop.__get__(self)
                elif isinstance(prop, property):
                    v = prop.fget(self)
                else:
                    v = getattr(self, k, _not_found_)
                if v is _not_found_:
                    v = self._entry.get(k)
                if v is _not_found_ or isinstance(v, Entry):
                    continue
                # elif hasattr(v, "__serialize__"):
                #     res[k] = v.__serialize__()
                # else:
                #     res[k] = serialize(v)
                res[k] = v
            return res

    @property
    def __category__(self):
        return super().__category__ | Node.Category.LIST

    def get(self, key: _TPath, default_value=_not_found_) -> _TObject:
        obj = self._entry.get(key)
        if isinstance(obj, Entry):
            obj = default_value
        return self.__post_process__(obj)

    def __post_process__(self, value: Any, *args, **kwargs) -> Any:
        return super().__post_process__(self.__new_child__(value), *args, **kwargs)

    def __getitem__(self, key: _TKey) -> _TObject:
        return Node.__getitem__(self, key)

    def __setitem__(self, key: _TKey, value: _TObject) -> None:
        Node.__setitem__(self, key, value)
        # if isinstance(key, str):
        #     self.__reset__([key])

    def __delitem__(self, key: _TKey) -> None:
        return Node.__delitem__(self, key)

    def __iter__(self) -> Iterator[_TObject]:
        yield from Node.__iter__(self)

    def __len__(self) -> int:
        return Node.__len__(self)

    def __eq__(self, o: object) -> bool:
        return Node.__eq__(self, o)

    def __contains__(self, o: object) -> bool:
        return Node.__contains__(self, o)

    def __ior__(self, other):
        if self._entry is None:
            self._entry = other
        elif isinstance(self._entry, Entry):
            self._entry.put(other)
        elif isinstance(self._entry, collections.abc.Mapping):
            for k, v in other.items():
                self.__setitem__(k, v)
        else:
            raise TypeError(f"{type(other)}")

    def __update__(self, d: Mapping) -> None:
        if not self._entry.writable:
            self._entry = Entry(d)  # Entry(collections.ChainMap(d, Dict[str, Node](self._entry)), writable=True)
        else:
            for k, v in d.items():
                self.__setitem__(k, v)

        self.__reset__(d.keys())

    def __reset__(self, d=None) -> None:
        if isinstance(d, str):
            return self.__reset__([d])
        elif d is None:
            return self.__reset__([d for k in dir(self) if not k.startswith("_")])
        elif isinstance(d, Mapping):
            properties = getattr(self.__class__, '_properties_', _not_found_)
            if properties is not _not_found_:
                data = {k: v for k, v in d.items() if k in properties}
            self._entry = Entry(data, parent=self._entry.parent)
            self.__reset__(d.keys())
        elif isinstance(d, Sequence):
            for key in d:
                if isinstance(key, str) and hasattr(self, key) and isinstance(getattr(self.__class__, key, _not_found_), functools.cached_property):
                    delattr(self, key)
