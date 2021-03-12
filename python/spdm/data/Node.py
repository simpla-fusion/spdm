import collections
import copy
import pprint

import numpy as np

from ..util.logger import logger
from .Entry import Entry


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

    class Mapping:
        def __init__(self, data=None, *args,  parent=None, **kwargs):
            self._parent = parent
            self._data = data or dict()

        def serialize(self):
            return {k: (v.serialize() if hasattr(v, "serialize") else v) for k, v in self.items()}

        @staticmethod
        def deserialize(cls, d, parent=None):
            res = cls(parent)
            res.update(d)
            return res

        def __repr__(self) -> str:
            return pprint.pformat(getattr(self, "_data", None))

        def __getitem__(self, key):
            return self._data[key]

        def __setitem__(self, key, value):
            self.insert(value, key)

        def __delitem__(self, k):
            del self._data[k]

        def __len__(self):
            return len(self._data)

        def __contain__(self, k):
            return k in self._data

        def __iter__(self):
            yield from self._data.values()

        def __merge__(self, d,  recursive=True):
            return NotImplemented

        def __ior__(self, other):
            return self.__merge__(other, recursive=True)

        """
            @startuml
            [*] --> Group

            Group       --> Mapping         : update(dict),insert(value,str),at(str)
            Group       --> Sequence        : update(list),insert(value,int),[_next_]

            Mapping     --> Mapping         : insert(value,key), at(key),
            Mapping     --> Sequence        : [_next_],

            Sequence    --> Sequence        : insert(value), at(int),
            Sequence    --> Illegal         : insert(value,str),get(str)

            Illegal     --> [*]             : Error

            @enduml
        """

        def update(self, other, *args, **kwargs):
            if other is None:
                return
            elif isinstance(other, collections.abc.Mapping):
                for k, v in other.items():
                    self.insert(v, k, *args, **kwargs)
            elif isinstance(other, collections.abc.Sequence):
                for v in other:
                    self.insert(v, None, *args, **kwargs)
            else:
                raise TypeError(f"Not supported operator! update({type(self)},{type(other)})")

        def insert(self, value, key=None, *args, **kwargs):
            res = self._data.get(key, None) or self._data.setdefault(
                key, self._parent.__new_node__(name=key, parent=self))
            res.__update__(value, *args, **kwargs)
            return res

    class Sequence:
        def __init__(self, data=None,   *args, parent=None, **kwargs) -> None:
            self._parent = parent
            self._data = data or list()

        def serialize(self):
            return [(v.serialize() if hasattr(v, "serialize") else v) for v in self]

        @staticmethod
        def deserialize(cls, d, parent=None):
            return cls(d, parent)

        def __repr__(self) -> str:
            return pprint.pformat(self._data)

        def __getitem__(self, key):
            return self._data[key]

        def __setitem__(self, key, value):
            self.insert(value, key)

        def __delitem__(self, k):
            del self._data[k]

        def __len__(self):
            return len(self._data)

        def __contain__(self, k):
            return k in self._data

        def __iter__(self):
            yield from self._data

        def insert(self, value, key=None, *args, **kwargs):
            if isinstance(key, int):
                res = self._data[key]
            else:
                self._data.append(self._parent.__new_node__(name=key, parent=self))
                res = self._data.__getitem__(-1)

            res.__update__(value, *args, **kwargs)
            return res

        def update(self, other, *args, **kwargs):
            if other is None:
                return
            if isinstance(other, collections.abc.Sequence):
                for v in other:
                    self.insert(v, *args, **kwargs)
            else:
                raise TypeError(f"Not supported operator! update({type(self)},{type(other)})")

    def __init__(self, value=None, *args,  name=None, parent=None, **kwargs):
        self._name = name  # or uuid.uuid1()
        self._parent = parent
        self._data = None

        if isinstance(value, Node):
            self._data = value._data
        else:
            value = self.__pre_process__(value)
            if isinstance(value, collections.abc.Mapping):
                self.__as_mapping__(value, force=True)
            elif isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
                self.__as_sequence__(value, force=True)
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

    def __as_mapping__(self, value=None,  force=True):
        if isinstance(self._data, Node.Mapping):
            self._data.update(value)
        elif self._data is None and force:
            self._data = self.__class__.Mapping(value, parent=self)
        else:
            raise ValueError(f"{type(self._data)} is not a Mapping!")
        return self._data

    def __as_sequence__(self, value=None, force=False):
        if isinstance(self._data, Node.Sequence):
            self._data.update(value)
        elif self._data is None and force:
            self._data = self.__class__.Sequence(value, parent=self)
        else:
            raise ValueError(f"{type(self._data)} is not a Sequence!")
        return self._data

    def __pre_process__(self, value, *args, **kwargs):
        return value

    def __post_process__(self, value, *args, **kwargs):
        if isinstance(value, (Node.Mapping, Node.Sequence, collections.abc.Mapping, collections.abc.Sequence, Entry)) and not isinstance(value, str):
            return self.__new_node__(value, parent=self)
        elif not isinstance(value, Node):
            return value
        else:
            raise TypeError(type(value))
        # elif isinstance(value._data, (collections.abc.Mapping, collections.abc.Sequence, type(None))):
        #     return self.__new_node__(value._data)
        # else:
        #     return value._data

    def __raw_set__(self, path, value):
        if isinstance(self._data, Entry):
            self._data.insert(path, value)
        elif path is None or len(path) == 0:
            self._data = value
        elif isinstance(path, str):
            self.__as_mapping__(force=True).insert(value, path)
        else:
            self.__as_sequence__(force=True).insert(value, path)

    def __raw_get__(self, path):
        if isinstance(path, slice):
            res = self.__fetch__()[path]
        elif isinstance(self._data, Entry):
            res = self._data.get(path)
        elif not path:
            res = self._data
        elif isinstance(path, str):
            res = self.__as_mapping__(force=False)[path]
        elif isinstance(path, _TAG_):
            res = self.__as_sequence__(force=True)[path]
        else:
            res = self.__as_sequence__(force=False)[path]

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
        elif not isinstance(self._data, (Node.Mapping, Node.Sequence)):
            raise TypeError(type(self._data))
        elif isinstance(path, str):
            del self._data[path]
        else:
            del self._data[path]

    def __contains__(self, path):
        if isinstance(self._data, Entry):
            return self._data.contains(path)
        elif isinstance(path, str):
            return path in self._data
        elif not isinstance(path, str):
            return path >= 0 and path < len(self._data)
        else:
            return False

    def __len__(self):
        if isinstance(self._data, Entry):
            return self._data.count()
        elif isinstance(self._data,  (Node.Mapping, Node.Sequence)) and not isinstance(self._data, str):
            return len(self._data)
        else:
            return 0 if self._data is None else 1

    def __iter__(self):
        if isinstance(self._data, (Node.Mapping, Node.Sequence, Entry)):
            yield from map(lambda v: self.__post_process__(v), self._data)
        else:
            yield self.__post_process__(self._data)

    def __update__(self, value, * args,   **kwargs):
        value = self.__pre_process__(value, *args, **kwargs)

        if isinstance(value, collections.abc.Mapping):
            self.__as_mapping__(value, force=True)
        elif isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
            self.__as_sequence__(value,  force=True)
        else:
            self._data = value

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
