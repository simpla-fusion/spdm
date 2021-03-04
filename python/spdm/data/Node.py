import collections
import uuid
from spdm.util.logger import logger


class _NEXT_TAG_:
    pass


_next_ = _NEXT_TAG_()
_last_ = -1


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

        def __missing__(self, key):
            obj = self._parent.__new_child__(name=key)
            super().__setitem__(key, obj)
            return obj

        def __setitem__(self, key, value):
            self.insert(value, key)

        def update(self, other, *args, **kwargs):
            if other is None:
                return
            elif isinstance(other, collections.abc.Mapping):
                for k, v in other.items():
                    self.insert(v, k)
            elif isinstance(other, collections.abc.Sequence):
                for v in other:
                    self.insert(v)
            else:
                raise TypeError(f"Not supported operator! update({type(self)},{type(other)})")

        def insert(self, value, key=None, *args, **kwargs):
            if key is None:
                obj = self.__new_child__(value, *args, name=key, **kwargs)
                super().__setitem__(obj.__name__, obj)
            else:
                self.at(key).__update__(value, *args, **kwargs)

        def at(self, key):
            return self.__getitem__(key)

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
                self.__getitem__(key).__update__(value, *args, **kwargs)
            else:
                self.append(self._parent.__new_child__(value, *args, name=key, **kwargs))

        def at(self, key):
            return self.__getitem__(key)

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
        return f"{self._value}"

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

    def __as_mapping__(self):
        if isinstance(self._value, Node.Mapping):
            return self._value
        elif not self._value:
            self._value = self.__class__.Mapping(self)
            return self._value
        else:
            raise RuntimeError(f"Can not convert {type(self._value)} to Mapping!")

    def __as_sequence__(self):
        if isinstance(self._value, Node.Sequence):
            return self._value
        else:
            res = self.__class__.Sequence(self)
            if not not self._value:
                self._value.insert(self._value)
            self._value = res
            return res

    def __new_child__(self, *args, parent=None, **kwargs):
        return self.__class__(*args,  parent=parent or self, **kwargs)

    def __update__(self, value, *args, **kwargs):
        if isinstance(value, collections.abc.Mapping):
            self.__as_mapping__().update(value, *args, **kwargs)
        elif isinstance(value, collections.abc.Sequence):
            self.__as_sequence__().update(value, *args, **kwargs)
        else:
            self._value = value

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.__as_mapping__().insert(value, key)
        else:
            self.__as_sequence__().insert(value, key)

    def __getitem__(self, key):
        if isinstance(key, str):
            res = self.__as_mapping__().at(key)
        else:
            res = self.__as_sequence__().at(key)

        return res if isinstance(res._value, (self.__class__.Sequence, self.__class__.Mapping, type(None))) else res._value
