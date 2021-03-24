import collections
import copy
import pprint
import functools
import collections
import numpy as np

from ..util.logger import logger
from .Entry import Entry, _next_, _last_, _not_found_, _NEXT_TAG_
from ..util.dict_util import deep_merge_dict


class Node:
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
    class LazyHolder:
        def __init__(self, parent, path, prefix=[]) -> None:
            self._parent = parent
            if isinstance(path, str):
                self._prefix = (prefix or []) + path.split('.')
            elif isinstance(path, collections.abc.MutableSequence):
                self._prefix = (prefix or []) + path
            else:
                self._prefix = (prefix or []) + [path]

        @property
        def parent(self):
            return self._parent

        @property
        def prefix(self):
            return self._prefix

        def extend(self, path):
            return self if path is None else Node.LazyHolder(self._parent, path, prefix=self._prefix)

    def __init__(self, data=None, *args,  parent=None, **kwargs):
        self._parent = parent
        self._data = data._data if isinstance(data, Node) else data

    def __repr__(self) -> str:
        return pprint.pformat(self._data) if not isinstance(self._data, str) else f"'{self._data}'"

    def __new_node__(self, *args, **kwargs):
        return self.__class__(*args,  **kwargs)

    def copy(self):
        if isinstance(Node, (Node.Mapping, Node.Sequence)):
            return self.__new_node__(self._data.copy())
        else:
            return self.__new_node__(copy.copy(self._data))

    # def entry(self, path, *args, **kwargs):
    #     if isinstance(self._data, Entry):
    #         return self._data.child(path, *args, **kwargs)
    #     else:
    #         return Entry(self._data, prefix=path, parent=self._parent)

    def serialize(self):
        return self._data.serialize() if hasattr(self._data, "serialize") else self._data

    @staticmethod
    def deserialize(cls, d):
        return cls(d)

    @property
    def __name__(self):
        return self._name

    def __fetch__(self):
        if isinstance(self._data, Entry):
            return self._data.get_value([])
        else:
            return self._data

    @property
    def __parent__(self):
        return self._parent

    def __hash__(self) -> int:
        return hash(self._name)

    def __clear__(self):
        self._data = None

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

    def __pre_process__(self, value, *args, **kwargs):
        return value

    def __post_process__(self, value, *args, **kwargs):
        if isinstance(value, (collections.abc.Mapping, collections.abc.MutableSequence)):
            if len(value) == 0:
                return None
            else:
                return self.__new_node__(value, parent=self)
        elif isinstance(value, (Entry, Node.LazyHolder)):
            return self.__new_node__(value, parent=self)
        else:
            return value

    def __raw_set__(self, key, value):
        if isinstance(self._data, Entry):
            return self._data.insert(key, value)
        elif key is None and self._data is None:
            self._data = value
            return self._data

        if isinstance(self._data, Node.LazyHolder):
            entry = self._data.extend(key)
            holder = entry.parent
            path = entry.prefix
        else:
            holder = self
            path = key

        if isinstance(path, str):
            path = path.split(".")
        elif not isinstance(path, collections.abc.MutableSequence):
            path = [path]

        if path is None or len(path) == 0:
            holder._data = value
            return

        if holder._data is None:
            if isinstance(path[0], str):
                holder._data = collections.OrderedDict()
            else:
                holder._data = list()

        obj = holder._data

        for idx, key in enumerate(path[:-1]):
            if isinstance(obj, collections.abc.Mapping):
                if not isinstance(key, str):
                    raise TypeError(f"mapping indices must be str, not {type(key).__name__}")
                obj = obj.setdefault(key, collections.OrderedDict() if isinstance(path[idx+1], str) else [])
            elif isinstance(obj, collections.abc.MutableSequence):
                if isinstance(key, _NEXT_TAG_):
                    obj.append({} if isinstance(path[idx+1], str) else [])
                    obj = obj[-1]
                elif isinstance(key, (int, slice)):
                    obj = obj[key]
                else:
                    raise TypeError(f"list indices must be integers or slices, not {type(key).__name__}")

            else:
                raise TypeError(f"Can not insert data to {path[:idx]}! type={type(obj)}")

        if isinstance(path[-1], _NEXT_TAG_):
            obj.append(value)
        else:
            obj[path[-1]] = value

    def __raw_get__(self, path):
        if isinstance(path, str):
            path = path.split(".")
        elif not isinstance(path, collections.abc.MutableSequence):
            path = [path]

        if isinstance(self._data, Entry):
            return self._data.get(path)
        elif path is None or (isinstance(path, collections.abc.MutableSequence) and len(path) == 0):
            return self._data
        elif isinstance(self._data, Node.LazyHolder):
            return self._data.extend(path)
        elif self._data is None:
            return Node.LazyHolder(self, path)

        obj = self._data

        for idx, key in enumerate(path):
            if obj is _not_found_:
                raise KeyError(f"{path[idx:]}")
            elif key is None or key == "":
                pass
            elif isinstance(obj, collections.abc.Mapping):
                if not isinstance(key, str):
                    raise TypeError(f"mapping indices must be str, not {type(key).__name__}! \"{key}\"")
                obj = obj.get(key, _not_found_)
            elif isinstance(obj, collections.abc.MutableSequence):
                if not isinstance(key, (int, slice)):
                    raise TypeError(f"list indices must be integers or slices, not {type(key).__name__}! \"{key}\"")
                elif isinstance(key, int) and key > len(self._data):
                    raise IndexError(f"Out of range! {key} > {len(self._data)}")
                obj = obj[key]
            else:
                obj = _not_found_

        if obj is _not_found_:
            return Node.LazyHolder(self, path)

        return obj

    def __setitem__(self, path, value):
        self.__raw_set__(path, self.__pre_process__(value))

    def __getitem__(self, path):
        return self.__post_process__(self.__raw_get__(path))

    def __delitem__(self, path):
        if isinstance(self._data, Entry):
            self._data.delete(path)
        elif not path:
            self.__clear__()
        elif not isinstance(self._data, (collections.abc.Mapping, collections.abc.MutableSequence)):
            raise TypeError(type(self._data))
        else:
            del self._data[path]

    def __contains__(self, path):
        if isinstance(self._data, Entry):
            return self._data.contains(path)
        elif isinstance(path, str) and isinstance(self._data, collections.abc.Mapping):
            return path in self._data
        elif not isinstance(path, str):
            return path >= 0 and path < len(self._data)
        else:
            return False

    def __len__(self):
        if isinstance(self._data, Entry):
            return self._data.count()
        elif isinstance(self._data,  (collections.abc.Mapping, collections.abc.MutableSequence)):
            return len(self._data)
        else:
            return 0 if self._data is None else 1

    def __iter__(self):
        if isinstance(self._data, (collections.abc.Mapping, collections.abc.MutableSequence, Entry)):
            yield from map(lambda v: self.__post_process__(v), self._data)
        else:
            yield self.__post_process__(self._data)

    def __ior__(self, other):
        if self._data is None:
            self._data = other
        elif isinstance(self._data, Node.LazyHolder):
            self.__raw_set__(None, other)
        elif isinstance(self._data, collections.abc.Mapping):
            for k, v in other.items():
                self.__setitem__(k, v)
        else:
            raise TypeError(f"{type(other)}")

    def __eq__(self, other) -> bool:
        if not isinstance(other, Node):
            other = self._data

        if isinstance(self._data, Entry):
            return self.equal(other)
        elif isinstance(self._data, Node.LazyHolder):
            return other is None
        else:
            return self._data == other
