import collections
import enum
import uuid
from numpy.lib.arraysetops import isin
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
import numpy as np


class _TAG_:
    pass


class _NEXT_TAG_(_TAG_):
    pass


class _LAST_TAG_(np.ndarray, _TAG_):
    def __init__(self) -> None:
        np.ndarray.__init__(self, -1)


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

    class Mapping(dict):
        def __init__(self, parent, *args,  **kwargs):
            super().__init__(*args, **kwargs)
            self._parent = parent

        def serialize(self):
            return {k: (v.serialize() if hasattr(v, "serialize") else v) for k, v in self.items()}

        @staticmethod
        def deserialize(cls, d, parent=None):
            res = cls(parent)
            res.update(d)
            return res

        def __iter__(self):
            for node in self.values():
                if hasattr(node, "__value__"):
                    yield node.__value__
                else:
                    yield node

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

        def __getitem__(self, key):
            return super().__getitem__(key)

        def __setitem__(self, key, value):
            self.insert(value, key)

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
            res = self.get(key, None) or self.setdefault(key, self._parent.__new_child__(name=key))
            res.__update__(value, *args, **kwargs)
            return res

        # def at(self, key):
        #     return self.__getitem__(key)

    class Sequence(list):
        def __init__(self, parent,  *args, **kwargs) -> None:
            super().__init__(self,  *args, **kwargs)
            self._parent = parent

        def serialize(self):
            return [(v.serialize() if hasattr(v, "serialize") else v) for v in self]

        @staticmethod
        def deserialize(cls, d, parent=None):
            res = cls(parent)
            res.update(d)
            return res

        def insert(self, value, key=None, *args, **kwargs):
            if isinstance(key, int):
                res = self.__getitem__(key)
            else:
                self.append(self._parent.__new_child__(name=key))
                res = self.__getitem__(-1)

            res.__update__(value, *args, **kwargs)
            return res

        # def at(self, key):
        #     if isinstance(key, (int, slice)):
        #         return self.__getitem__(key)
        #     else:
        #         raise KeyError(key)

        def update(self, other, *args, **kwargs):
            if isinstance(other, collections.abc.Sequence):
                for v in other:
                    self.insert(v, *args, **kwargs)
            else:
                raise TypeError(f"Not supported operator! update({type(self)},{type(other)})")

    def __init__(self, value=None, *args,  name=None, parent=None, **kwargs):
        self._name = name or uuid.uuid1()
        self._parent = parent
        self._value = None
        self.__update__(value, *args, **kwargs)

    def __repr__(self) -> str:
        return f"{self._value}" if not isinstance(self._value, str) else f"'{self._value}'"

    def serialize(self):
        return self._value.serialize() if hasattr(self._value, "serialize") else self._value

    @staticmethod
    def deserialize(cls, d):
        return cls(d)

    @property
    def __name__(self):
        return self._name

    @property
    def __value__(self):
        return self._value

    @property
    def __parent__(self):
        return self._parent

    @property
    def __metadata__(self):
        return self._metadata

    def __hash__(self) -> int:
        return hash(self._name)

    def __clear__(self):
        self._value = None

    """
        @startuml
        [*] --> Empty
        Empty       --> Sequence        : as_sequence, __update__(list), __setitem__(int,v),__getitem__(int)
        Empty       --> Mapping         : as_mapping , __update__(dict), __setitem__(str,v),__getitem__(str)
        Empty       --> Empty           : clear


        Item        --> Item            : "__value__"
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

    def __as_mapping__(self, force=False):
        if isinstance(self._value, Node.Mapping):
            return self._value
        elif not self._value and force:
            self._value = self.__class__.Mapping(self)
            return self._value
        else:
            raise ValueError(f"{type(self._value)} is not a Mapping!")

    def __as_sequence__(self, force=False):
        if isinstance(self._value, Node.Sequence):
            return self._value
        elif force:
            res = self.__class__.Sequence(self)
            if not not self._value:
                self._value.insert(self._value)
            self._value = res
            return res

    def __new_child__(self, *args, parent=None, **kwargs):
        return self.__class__(*args,  parent=parent or self, **kwargs)

    def __update__(self, value, *args, **kwargs):
        value = self.__pre_process__(value, *args, **kwargs)
        if isinstance(value, collections.abc.Mapping):
            self.__as_mapping__(True).update(value, *args, **kwargs)
        elif isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
            self.__as_sequence__(True).update(value, *args, **kwargs)
        else:
            self._value = value

    def __pre_process__(self, value, *args, **kwargs):
        return value

    def __post_process__(self, value, *args, **kwargs):
        if not isinstance(value, Node) or isinstance(value._value, (self.__class__.Sequence, self.__class__.Mapping, type(None))):
            return value
        else:
            return value._value

    def __setitem__(self, path, value):
        if isinstance(path, str):
            path = path.split('.')
        elif not isinstance(path, list):
            path = [path]

        obj = self

        for k in path:
            if isinstance(k, str):
                obj = obj.__as_mapping__(True).insert(None, k)
            else:
                obj = obj.__as_sequence__(True).insert(None, k)

        obj.__update__(value)

    def __missing__(self, path):
        return LazyProxy(self, prefix=path)

    def __getitem__(self, path):
        if isinstance(path, str):
            path = path.split('.')
        elif not isinstance(path, list):
            path = [path]

        obj = self
        for i, key in enumerate(path):
            try:
                if isinstance(key, str):
                    res = obj.__as_mapping__(False)[key]
                elif isinstance(key, _TAG_):
                    res = obj.__as_sequence__(True)[key]
                else:
                    res = obj.__as_sequence__(False)[key]
            except Exception:
                obj = obj.__missing__(path[i:])
                break
            else:
                obj = res

        return self.__post_process__(obj)

    def __delitem__(self, key):
        if isinstance(key, str) and isinstance(self._value, collections.abc.Mapping):
            del self._value[key]
        elif not isinstance(key, str) and isinstance(self._value, collections.abc.Sequence):
            del self._value[key]
        else:
            raise RuntimeError((type(self._value), type(key)))

    def __contain__(self, key):
        if isinstance(key, str) and isinstance(self._value, collections.abc.Mapping):
            return key in self._value
        elif not isinstance(key, str) and isinstance(self._value, collections.abc.Sequence):
            return key >= 0 and key < len(self._value)
        else:
            return False

    def __len__(self):
        if isinstance(self._value, collections.abc.Mapping):
            return len(self._value)
        elif isinstance(self._value, collections.abc.Sequence) and not isinstance(self._value, str):
            return len(self._value)
        else:
            return 0 if self._value is None else 1

    def __iter__(self):
        if isinstance(self._value, collections.abc.Mapping):
            yield from map(lambda v: self.__post_process__(v), self._value.values())
        elif isinstance(self._value, collections.abc.Sequence) and not isinstance(self._value, str):
            yield from map(lambda v: self.__post_process__(v), self._value)
        else:
            yield self._value

    class __lazy_proxy__:
        @staticmethod
        def put(self, path, value):
            self.__setitem__(path, value)

        @staticmethod
        def get(self, path):
            res = self.__getitem__(path)
            if isinstance(res, LazyProxy):
                raise KeyError(path)
            else:
                return res

        # @staticmethod
        # def iter(self, path):
        #     obj = self[path]
        #     logger.debug(obj)
            # yield obj
            # if isinstance(path, str):
            #     path = path.split('.')

            # obj = self
            # for i, k in enumerate(path):
            #     if isinstance(k, str):
            #         obj = obj.__as_mapping__()
            #     try:
            #         obj = obj[k]
            #     except KeyError:
            #         raise KeyError('.'.join(map(str, path[:i+1])))

            # return None

        # def delete(self, key):
        #     if isinstance(self.__data__, collections.abc.Mapping) or isinstance(self.__data__, collections.abc.Sequence):
        #         try:
        #             del self.__data__[key]
        #         except KeyError:
        #             pass

        # def contain(self, key):
        #     if self.__data__ is None:
        #         return False
        #     elif isinstance(key, str):
        #         return key in self.__data__
        #     elif type(key) is int:
        #         return key < len(self.__data__)
        #     else:
        #         raise KeyError(key)

        # def count(self):
        #     if self.__data__ is None:
        #         return 0
        #     else:
        #         return len(self.__data__)

        # def iter(self):
        #     if self.__data__ is None:
        #         return
        #     elif isinstance(self.__data__,  collections.abc.Mapping):
        #         for k, v in self.__data__.items():
        #             if isinstance(v, collections.abc.Mapping):
        #                 v = AttributeTree(v)

        #             yield k, v
        #     else:
        #         for v in self.__data__:
        #             if isinstance(v, collections.abc.Mapping):
        #                 v = AttributeTree(v)
        #             yield v
