import collections
import copy
import pprint
from typing import Type

import numpy as np

from ..util.logger import logger
from .Entry import Entry
from ..util.dict_util import deep_merge_dict


class _TAG_:
    pass


class _NEXT_TAG_(_TAG_):
    pass


class _LAST_TAG_(np.ndarray, _TAG_):
    @staticmethod
    def __new__(cls,   *args,   **kwargs):
        return np.asarray(-1).view(cls)

    def __init__(self, *args, **kwargs) -> None:
        pass


_next_ = _NEXT_TAG_()
_last_ = _LAST_TAG_()


class Node:
    """
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

    def __init__(self, value=None, *args,  name=None, parent=None, **kwargs):
        self._name = name  # or uuid.uuid1()
        self._parent = parent
        self._data = None

        if isinstance(value, Node):
            self._data = value._data
        else:
            value = self.__pre_process__(value)
            if isinstance(value, collections.abc.Mapping):
                self.__as_mapping__(value)
            elif isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
                self.__as_sequence__(value)
            else:
                self._data = value

    def __repr__(self) -> str:
        v = self.__fetch__()
        return pprint.pformat(v) if not isinstance(v, str) else f"'{v}'"

    def __new_node__(self, *args, **kwargs):
        return self.__class__(*args,  **kwargs)

    def copy(self):
        if isinstance(Node, (Node.Mapping, Node.Sequence)):
            return self.__new_node__(self._data.copy())
        else:
            return self.__new_node__(copy.copy(self._data))

    def entry(self, path, *args, **kwargs):
        if isinstance(self._data, Entry):
            return self._data.child(path, *args, **kwargs)
        else:
            return Entry(self._data, prefix=path, parent=self._parent)

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

    def __normalize_path__(self, path=None):
        if path is None:
            pass
        elif isinstance(path, str):
            path = path.split(".")
        elif not isinstance(path, collections.abc.Sequence):
            path = [path]
        return path

    def __as_mapping__(self, value=None):
        if isinstance(value, collections.abc.Mapping):
            if self._data is None:
                self._data = {}
            self._data = deep_merge_dict(self._data, value)
        elif value is not None:
            raise TypeError(f"{type(value)} is not a Mapping!")
        elif self._data is None:
            self._data = {}
        elif not isinstance(self._data, collections.abc.Mapping):
            raise ValueError(f"{type(self._data)} is not a Mapping!")
        return self._data

    def __as_sequence__(self, value=None, force=False):
        if isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
            self._data = value
        elif value is not None:
            raise TypeError(f"{type(value)} is not a Sequence!")
        elif self._data is None:
            self._data = list()
        elif not isinstance(self._data, collections.abc.Sequence):
            raise ValueError(f"{type(self._data)} is not a Sequence!")
        return self._data

    def __pre_process__(self, value, *args, **kwargs):
        return value

    def __post_process__(self, value, *args, **kwargs):
        if isinstance(value, str):
            return value
        elif isinstance(value, (collections.abc.Mapping, collections.abc.Sequence, Entry)):
            return self.__new_node__(value, parent=self)
        else:
            return value

    def __raw_set__(self, path, value):
        if isinstance(self._data, Entry):
            self._data.insert(path, value)
        elif path is None or len(path) == 0:
            self._data = value
        elif isinstance(path, _NEXT_TAG_):
            self.__as_sequence__().append(value)
        elif not isinstance(path, str):
            self.__as_sequence__()[path] = value
        else:
            self.__as_mapping__()[path] = value

    def __raw_get__(self, path):
        if isinstance(self._data, Entry):
            res = self._data.get(path)
        elif isinstance(path, str):
            res = self.__as_mapping__()[path]
        elif path is None and (isinstance(path, list) and len(path) == 0):
            res = self._data
        elif isinstance(path, (int, slice)):
            res = self.__as_sequence__()[path]
        elif isinstance(path, _NEXT_TAG_):
            self.__as_sequence__().append({})
            res = self.__as_sequence__()[-1]

        else:
            raise TypeError(type(path))

        return res

    def __setitem__(self, path, value):
        self.__raw_set__(path, self.__pre_process__(value))

    def __getitem__(self, path):
        return self.__post_process__(self.__raw_get__(path))

    def __delitem__(self, path):
        if isinstance(self._data, Entry):
            self._data.delete(path)
        elif not path:
            self.__clear__()
        elif isinstance(self._data, str):
            self._data = None
        elif not isinstance(self._data, (collections.abc.Mapping, collections.abc.Sequence)):
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
        elif isinstance(self._data, str):
            return 1
        elif isinstance(self._data,  (collections.abc.Mapping, collections.abc.Sequence)):
            return len(self._data)
        else:
            return 0 if self._data is None else 1

    def __iter__(self):
        if isinstance(self._data, (collections.abc.Mapping, collections.abc.Sequence, Entry)):
            yield from map(lambda v: self.__post_process__(v), self._data)
        else:
            yield self.__post_process__(self._data)

    # def __update__(self, value, * args,   **kwargs):
    #     value = self.__pre_process__(value, *args, **kwargs)

    #     if isinstance(value, collections.abc.Mapping):
    #         self.__as_mapping__(value)
    #     elif isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
    #         self.__as_sequence__(value)
    #     else:
    #         self._data = value

    def __ior__(self, other):
        if self._data is None:
            self._data = other
        elif not isinstance(self._data, collections.abc.Mapping):
            self._data |= other
        elif isinstance(self._data, collections.abc.Mapping):
            for k, v in other.items():
                self.__setitem__(k, v)
        else:
            raise TypeError(f"{type(other)}")

    class __lazy_proxy__:
        @staticmethod
        def put(self, path, value):
            self.__setitem__(path, value)

        @staticmethod
        def get(self, path):
            return self.__getitem__(path)

        @staticmethod
        def count(self, path):
            return len(self.__getitem__(path))
