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

import numpy as np, scipy
from ..util.dict_util import deep_merge_dict
from ..util.logger import logger
from ..util.sp_export import sp_find_module
from ..util.utilities import _not_found_, _undefined_, serialize
from .Entry import (_DICT_TYPE_, _LIST_TYPE_, Entry, EntryCombiner,
                    EntryContainer, _next_, _TKey, _TObject, _TPath)

_TNode = TypeVar('_TNode', bound='Node')

_T = TypeVar("_T")


class Node(EntryContainer, Generic[_TObject]):
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

    def __init__(self, entry: Any = None, /, parent=None, new_child=_undefined_) -> None:
        super().__init__(entry)
        self._parent = parent
        self._new_child = new_child

    def __repr__(self) -> str:
        annotation = [f"{k}='{v}'" for k, v in self.annotation.items() if v is not None]
        return f"<{getattr(self,'__orig_class__',self.__class__.__name__)} {' '.join(annotation)}/>"

    @property
    def annotation(self) -> dict:
        return {
            "id": self.nid,
            "type":  self._entry.__class__.__name__
        }

    @property
    def nid(self) -> str:
        return self.get("@id", None)

    def _attribute_type(self, attribute=_undefined_):
        attr_type = _undefined_

        if isinstance(attribute, str):
            attr = dict(inspect.getmembers(self.__class__)).get(attribute, _not_found_)
            if isinstance(attr, (_sp_property, cached_property)):
                attr_type = attr.func.__annotations__.get("return", None)
            elif isinstance(attr, (property)):
                attr_type = attr.fget.__annotations__.get("return", None)
        elif attribute is _undefined_:
            child_cls = Node
            #  @ref: https://stackoverflow.com/questions/48572831/how-to-access-the-type-arguments-of-typing-generic?noredirect=1
            orig_class = getattr(self, "__orig_class__", None)
            if orig_class is not None:
                child_cls = get_args(self.__orig_class__)
                if child_cls is not None and len(child_cls) > 0 and inspect.isclass(child_cls[0]):
                    child_cls = child_cls[0]
            attr_type = child_cls
        else:
            raise NotImplementedError(attribute)

        return attr_type

    def _convert(self, value: _T, *args,  attribute=_undefined_, parent=_undefined_, **kwargs) -> Union[_T, _TObject]:
        if parent is _undefined_:
            parent = self

        if isinstance(value, Entry.PRIMARY_TYPE) or value in (None, _not_found_, _undefined_):
            return value
        # elif (isinstance(value, list) and all(filter(lambda d: isinstance(d, (int, float, np.ndarray)), value))):
        #     return value
        elif inspect.isclass(self._new_child):
            if isinstance(value, self._new_child):
                return value
            elif issubclass(self._new_child, Node):
                return self._new_child(value, parent=parent, **kwargs)
            else:
                return self._new_child(value, **kwargs)
        elif callable(self._new_child):
            return self._new_child(value, **kwargs)
        elif isinstance(self._new_child, collections.abc.Mapping) and len(self._new_child) > 0:
            kwargs = collections.ChainMap(kwargs, self._new_child)
        elif self._new_child is not _undefined_ and not not self._new_child:
            logger.warning(f"Ignored!  { (self._new_child)}")

        if isinstance(attribute, str) or attribute is _undefined_:
            attribute_type = self._attribute_type(attribute)
        else:
            attribute_type = attribute

        if inspect.isclass(attribute_type):
            if isinstance(value, attribute_type):
                res = value
            elif attribute_type in (int, float):
                res = attribute_type(value)
            elif attribute_type is np.ndarray:
                res = np.asarray(value)
            elif dataclasses.is_dataclass(attribute_type):
                if isinstance(value, collections.abc.Mapping):
                    res = attribute_type(**{k: value.get(k, None) for k in attribute_type.__dataclass_fields__})
                elif isinstance(value, collections.abc.Sequence):
                    res = attribute_type(*value)
                else:
                    res = attribute_type(value)
            elif issubclass(attribute_type, Node):
                res = attribute_type(value, parent=parent, **kwargs)
            else:
                res = attribute_type(value, **kwargs)
        elif hasattr(attribute_type, '__origin__'):
            if issubclass(attribute_type.__origin__, Node):
                res = attribute_type(value, parent=parent, **kwargs)
            else:
                res = attribute_type(value, **kwargs)
        elif callable(attribute_type):
            res = attribute_type(value, **kwargs)
        elif attribute_type is not _undefined_:
            raise TypeError(attribute_type)
        elif isinstance(value, collections.abc.Sequence):
            res = List(value, parent=self, **kwargs)
        elif isinstance(value, collections.abc.Mapping):
            res = Dict(value, parent=self, **kwargs)
        elif isinstance(value, Entry):
            res = Node(value, parent=self, **kwargs)

        return res

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

    def _duplicate(self, *args, parent=None,  **kwargs) -> _TNode:
        return super()._duplicate(*args, parent=parent if parent is not None else self._parent, **kwargs)

    def __hash__(self) -> int:
        return NotImplemented

    def _pre_process(self, value: _T, *args, **kwargs) -> _T:
        return value

    def _post_process(self, value: _T,   *args, path: _TPath = None, **kwargs) -> Union[_T, _TObject]:
        return self._convert(value, *args, **kwargs)

    def fetch(self, path: _TPath = None) -> _TObject:
        path = Entry.normalize_path(path)
        target = self
        val = _not_found_

        for key in path:
            val = _not_found_
            if isinstance(key, str):
                val = getattr(target, key, _not_found_)
            if val is not _not_found_:
                target = val
            else:
                target = target[key]

        return target

    def __eq__(self, other) -> bool:
        return self.equal([], other)

    # class Category(IntFlag):
    #     UNKNOWN = 0
    #     ITEM = 0x000
    #     DICT = 0x100
    #     LIST = 0x200
    #     ENTRY = 0x400
    #     ARRAY = 0x010
    #     INT = 0x001
    #     FLOAT = 0x002
    #     COMPLEX = 0x004
    #     STRING = 0x008

    # @staticmethod
    # def __type_category__(d) -> IntFlag:
    #     flag = Node.Category.UNKNOWN
    #     if hasattr(d,  "__array__"):
    #         flag |= Node.Category.ARRAY
    #         # if np.issubdtype(d.dtype, np.int64):
    #         #     flag |= Node.Category.INT
    #         # elif np.issubdtype(d.dtype, np.float64):
    #         #     flag |= Node.Category.FLOAT
    #     elif isinstance(d, collections.abc.Mapping):
    #         flag |= Node.Category.DICT
    #     elif isinstance(d, collections.abc.Sequence):
    #         flag |= Node.Category.LIST
    #     elif isinstance(d, int):
    #         flag |= Node.Category.INT
    #     elif isinstance(d, float):
    #         flag |= Node.Category.FLOAT
    #     elif isinstance(d, str):
    #         flag |= Node.Category.STRING
    #     # if isinstance(d, (Entry)):
    #     #     flag |= Node.Category.ENTRY

    #     return flag

    # @property
    # def __category__(self) -> Category:
    #     return Node.__type_category__(self._entry)


class List(Node[_TObject], Sequence[_TObject]):
    __slots__ = ("_common_kwargs")

    def __init__(self, cache: Union[Sequence, Entry] = None, /, parent=_undefined_, new_child=_undefined_,  **kwargs) -> None:
        if cache is None:
            cache = _LIST_TYPE_()
        elif not isinstance(cache, (Entry, list)):
            cache = [cache]
        Node.__init__(self, cache, parent=parent, new_child=new_child)
        self._common_kwargs = kwargs

    def _serialize(self) -> Sequence:
        return [serialize(v) for v in self._as_list()]

    @classmethod
    def _deserialize(cls, desc: Any) -> _TNode:
        return NotImplemented

    def _convert(self, value: _T,  parent=_undefined_, **kwargs) -> Union[_T, _TObject]:
        if parent is _undefined_:
            parent = self._parent
        return super()._convert(value, parent=parent, **collections.ChainMap(kwargs, self._common_kwargs))

    def _post_process(self, value: _T,   *args, path: Union[int, str] = _undefined_, **kwargs) -> _TObject:
        n_value = super()._post_process(value, *args, **kwargs)
        if n_value is value or (isinstance(value, Entry) and value.level > 0):
            pass
        elif isinstance(path, int) or (isinstance(path, list) and len(path) == 1 and isinstance(path[0], int)):
            n_value = self.replace(path, n_value)
        return n_value

    @property
    def _is_list(self) -> bool:
        return True

    def __len__(self) -> int:
        return self._entry.count()

    def __setitem__(self, path: _TPath, v: _T) -> None:
        super().__setitem__(path,  v)

    def __getitem__(self, path: _TPath) -> _TObject:
        return super().__getitem__(path)

    def __delitem__(self, path: _TPath) -> None:
        return super().__delitem__(path)

    def __iter__(self) -> Iterator[_TObject]:
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)

    def __iadd__(self, other):
        self._entry.put(_next_, other)
        return self

    def sort(self):
        if hasattr(self._entry.__class__, "sort"):
            self._entry.sort()
        else:
            raise NotImplementedError()

    def combine(self, default_value=None, predication=_undefined_, reducer=_undefined_, partition=_undefined_) -> _TObject:
        if predication is not _undefined_:
            target = self.get([], predication=predication)
        else:
            target = self
        return self._post_process(EntryCombiner(target, default_value=default_value,  reducer=reducer, partition=partition))

    def reset(self, value=None):
        if isinstance(value, (collections.abc.Sequence)):
            super().reset(value)
        else:
            self._combine = value
            super().reset()

    def find(self, predication,  only_first=True) -> _TObject:
        return self._post_process(self._entry.pull(predication=predication, only_first=only_first))

    def update(self, d, predication=_undefined_, only_first=False) -> int:
        return self._entry.push([], self._pre_process(d), predication=predication, only_first=only_first)


class Dict(Node[_TObject], Mapping[str, _TObject]):
    __slots__ = ()

    def __init__(self, cache: Optional[Mapping] = None,  /, parent=None, new_child=None,  **kwargs):

        if cache is None:
            cache = _DICT_TYPE_()

        if len(kwargs) > 0:
            if isinstance(cache, collections.abc.Mapping):
                deep_merge_dict(cache, kwargs)
            else:
                logger.warning(f"ignore kwargs: {kwargs.keys()}")
                raise RuntimeError(kwargs.keys())

        Node.__init__(self, cache,  parent=parent, new_child=new_child)

    @property
    def _is_dict(self) -> bool:
        return True

    def _serialize(self) -> Mapping:
        return {k: serialize(v) for k, v in self._as_dict()}

    @classmethod
    def _deserialize(cls, desc: Any) -> _TNode:
        return NotImplemented

    def _post_process(self, value: _T,   *args, path: Union[int, str] = _undefined_, **kwargs) -> Union[_T, _TObject]:
        n_value = super()._post_process(value, *args, **kwargs)

        if n_value is value or (isinstance(value, Entry) and value.level > 0):
            pass
        elif isinstance(path, str) or (isinstance(path, list) and len(path) == 1 and isinstance(path[0], str)):
            n_value = self.replace(path, n_value)

        return n_value

    def __getitem__(self, path: _TPath) -> _TObject:
        return super().__getitem__(path)

    def __setitem__(self, key: str, value: _T) -> None:
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError()

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o)

    def __contains__(self, o: object) -> bool:
        return super().__contains__(o)

    def __ior__(self, other) -> _TNode:
        return self.put(None, {Entry.op_tag.update: other})

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


class _sp_property(Generic[_T]):

    def __init__(self, func: Callable[..., _T]):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.lock = RLock()
        self.return_type = func.__annotations__.get("return", None)

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def _check_type(self, value):
        orig_class = getattr(value, "__orig_class__", value.__class__)
        return self.return_type is None \
            or orig_class == self.return_type \
            or (inspect.isclass(orig_class)
                and inspect.isclass(self.return_type)
                and issubclass(orig_class, self.return_type))

    def _convert(self, instance: Node, value: _T) -> _T:
        if self._check_type(value):
            n_value = value
        elif hasattr(instance, "_convert"):
            n_value = instance._convert(value, attribute=self.return_type)
        else:
            n_value = self.return_type(value)

        return n_value

    def _get_entry(self, instance: Node) -> Entry:
        try:
            entry = getattr(instance, "_entry", _not_found_)
            if entry is _not_found_:
                entry = Entry(instance.__dict__)
        except AttributeError as error:
            logger.exception(error)
            raise AttributeError(error)

        return entry

    def __set__(self, instance: Node, value: Any):
        if instance is None:
            return self
        with self.lock:
            if self._check_type(value):
                if value._parent is None:
                    value._parent = instance
                else:
                    value = value._duplicate(parent=instance)
                self._get_entry(instance).put(self.attrname, value)
            else:
                self._get_entry(instance).put(self.attrname, self._convert(instance, value))

    def __get__(self, instance: Node, owner=None) -> _T:
        if instance is None:
            return self

        if self.attrname is None:
            raise TypeError("Cannot use sp_property instance without calling __set_name__ on it.")

        with self.lock:
            entry = self._get_entry(instance)

            value = entry.get(self.attrname, _not_found_)

            if not self._check_type(value):
                n_value = self._convert(instance,  self.func(instance))

                entry.replace(self.attrname, n_value)
            else:
                n_value = value

            return n_value

        return n_value

    def __delete__(self, instance: Node) -> None:
        if instance is None:
            return
        with self.lock:
            entry = self._get_entry(instance)
            entry.remove(self.attrname)


def sp_property(func: Callable[..., _T]) -> _sp_property[_T]:
    return _sp_property[_T](func)


def chain_map(*args, **kwargs) -> collections.ChainMap:
    d = []
    for a in args:
        if isinstance(d, collections.abc.Mapping):
            d.append(a)
        elif isinstance(d, Entry):
            d.append(Dict(a))
    return collections.ChainMap(*d, **kwargs)
