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

from ..numlib import np, scipy
from ..util.logger import logger
from ..util.sp_export import sp_find_module
from ..util.utilities import _not_found_, _undefined_, serialize
from .Entry import (_DICT_TYPE_, _LIST_TYPE_, Entry, EntryCombiner,
                    EntryContainer, _next_, _TKey, _TObject, _TQuery)

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

    def __init__(self, entry: Any = None, /, parent=None, new_child=_undefined_, **kwargs):

        super().__init__(entry)

        self._parent = parent

        if new_child is _undefined_:
            new_child = kwargs
        elif len(kwargs) > 0:
            logger.warning(f"Ignore kwargs: {kwargs}")

        self._new_child = new_child

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

    def _attribute_type(self, attribute=_undefined_) -> _T:
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

    def _duplicate(self) -> _TNode:
        obj = super()._duplicate()
        obj._parent = self._parent
        return obj

    def __hash__(self) -> int:
        return NotImplemented

    def _pre_process(self, value: Any, *args, **kwargs) -> Any:
        return value

    def _post_process(self, value: _T,   *args, query=_undefined_, **kwargs) -> Union[_T, _TNode]:
        return self._convert(value, *args, **kwargs)

    def fetch(self, query: _TQuery = None,  **kwargs) -> _T:
        query = Entry.normalize_query(query)
        target = self
        val = _not_found_

        for key in query:
            val = _not_found_
            if isinstance(key, str):
                val = getattr(target, key, _not_found_)
            if val is not _not_found_:
                target = val
            else:
                target = target[key]

        return target

    def bind(self, query: _TQuery, n_cls: _T) -> _T:
        return NotImplemented

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


class List(Node[_T], Sequence[_T]):
    __slots__ = ()

    def __init__(self, cache: Union[Sequence, Entry] = None, /, parent=_undefined_,   **kwargs) -> None:
        if cache is None:
            cache = _LIST_TYPE_()
        # elif cache.__class__ is Entry:
        #     tmp = cache.pull(_not_found_)
        #     if tmp is _not_found_:
        #         tmp = _LIST_TYPE_()
        #         cache.push(tmp)
        #     cache = tmp

        elif not isinstance(cache, (Entry, list)):
            cache = [cache]

        Node.__init__(self, cache, parent=parent, **kwargs)

    def _serialize(self) -> Sequence:
        return [serialize(v) for v in self._as_list()]

    @classmethod
    def _deserialize(cls, desc: Any) -> _TNode:
        return NotImplemented

    def _convert(self, value: _T,  parent=_undefined_, **kwargs) -> Union[_T, _TObject]:
        if parent is _undefined_:
            parent = self._parent
        return super()._convert(value, parent=parent, **kwargs)

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
        self._entry = Entry([self._post_process(d) for d in self._entry.iter()])
        for element in self._entry.iter():
            if hasattr(element.__class__, 'refresh'):
                element.refresh(**kwargs)

    def reset(self, value=None):
        if isinstance(value, (collections.abc.Sequence)):
            super().reset(value)
        else:
            self._combine = value
            super().reset()

    def find(self, predication,  only_first=True) -> _T:
        return self._post_process(self._entry.find(predication, only_first=only_first))

    def update(self, d, predication=_undefined_, only_first=False) -> int:
        return self._entry.update(self._pre_process(d), predication=predication, only_first=only_first)


class Dict(Node[_T], Mapping[str, _T]):
    __slots__ = ()

    def __init__(self, cache: Optional[Mapping] = None,  /,  **kwargs):
        # if cache.__class__ is Entry:
        #     tmp = cache.pull(_not_found_)
        #     if tmp is _not_found_:
        #         tmp = __class__()
        #         cache.push(tmp)
        #     cache = tmp
        if cache is None:
            cache = _DICT_TYPE_()

        Node.__init__(self, cache,   **kwargs)

    def _serialize(self) -> Mapping:
        return {k: serialize(v) for k, v in self._as_dict()}

    @ classmethod
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

    def _isinstance(self, obj) -> bool:
        res = True
        if self.return_type is not None:
            orig_class = getattr(obj, "__orig_class__", obj.__class__)
            res = orig_class == self.return_type \
                or (inspect.isclass(orig_class)
                    and inspect.isclass(self.return_type)
                    and issubclass(orig_class, self.return_type))

        return res

    def __set__(self, instance: Any, value: Any):
        with self.lock:
            cache = getattr(instance, "_entry", None)
            if cache is None:
                cache = Entry(instance.__dict__)
            cache.insert(self.attrname, value, op=Entry.op_tag.assign)

    def __get__(self, instance: Node, owner=None) -> _T:
        if instance is None:
            return self
        try:
            cache = getattr(instance, "_entry", None)
            if cache is None:
                cache = Entry(instance.__dict__)
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
                        val = instance._convert(val, attribute=self.return_type)

                    cache.put(self.attrname, val)

        return val


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
