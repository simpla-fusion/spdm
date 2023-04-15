from __future__ import annotations

import collections
import collections.abc
import dataclasses
import functools
import inspect
import typing

import numpy as np

from ..common.tags import _not_found_, _undefined_
from ..util.logger import logger
from ..util.misc import as_dataclass
from .Entry import Entry, as_entry
from .Function import Function
from .Node import Node
from .Path import Path
from .sp_property import sp_property
_TKey = typing.TypeVar("_TKey")
_TObject = typing.TypeVar("_TObject")
_T = typing.TypeVar("_T")


class Container(Node, typing.Container[_TObject]):
    r"""
       Container Node
    """

    def __init__(self, *args,   cache_data=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {} if cache_data is None else cache_data

    def __serialize__(self) -> dict:
        return self._entry.__serialize__()

    def duplicate(self) -> Container:
        other: Container = super().duplicate()  # type:ignore
        other._cache = self._cache
        return other

    def update(self, value) -> typing.Any:
        if isinstance(value, collections.abc.Mapping):
            return self._cache.update(value)
        else:
            raise TypeError(f"Invalid type {type(value)}")

    def __setitem__(self, path, value) -> typing.Any:
        path = Path(path)
        parent = self.get(path[:-1], parents=True)
        if isinstance(parent, Container):
            return parent._cache.__setitem__(path[-1], value)
        else:
            return as_entry(parent).insert(path[-1], value)

    def get(self, path, default_value=_not_found_, **kwargs):
        return Container._get(self, Path(path), default_value=default_value,   **kwargs)

    @staticmethod
    def _get(obj, path: list, default_value=_not_found_,   **kwargs):

        for idx, query in enumerate(path[:]):
            if not isinstance(obj, Container):
                obj = as_entry(obj).child(path[idx:], force=True).query(defalut_value=default_value, **kwargs)
                break
            elif isinstance(query, set):
                obj = {k: Container._get(obj, [k] + path[idx+1:], **kwargs) for k in query}
                break
            elif isinstance(query, tuple):
                obj = tuple([Container._get(obj, [k] + path[idx+1:], **kwargs) for k in query])
                break
            elif isinstance(query, dict) and isinstance(obj, collections.abc.Sequence):
                only_first = kwargs.get("only_first", False) or query.get("@only_first", True)
                if only_first:
                    obj = obj._as_child(None, obj._entry.child(query))
                else:
                    other: Container = Container.__new__(obj.__class__)  # type:ignore
                    other._entry = obj._entry.child(query)
                    other._cache = {}
                    obj = other
                continue
            elif isinstance(query,  slice) and isinstance(obj, collections.abc.Sequence):
                obj = obj.duplicate()
                obj._entry = obj._entry.child(query)
                continue
            elif isinstance(query, (str, int)):
                obj = obj._as_child(query, **kwargs)
                continue
            else:
                raise TypeError(f"Invalid key type {type(query)}")

        if obj is not _not_found_:
            pass
        elif default_value is _not_found_:
            raise KeyError(f"Key {path} not found")
        else:
            obj = default_value

        return obj

    def __getitem__(self, path) -> _TObject:
        return Container._get(self, Path(path), default_value=_not_found_)  # type:ignore

    def __delitem__(self, key) -> bool:
        return self._entry.child(key).remove() > 0

    def __contains__(self, key) -> bool:
        return self._entry.child(key).exists

    def __eq__(self, other) -> bool:
        return self._entry.equal(other)

    def __len__(self) -> int:
        return self._entry.count

    def _as_child(self,  
                  key: typing.Union[int, str, None],
                  value=_not_found_,
                  type_hint: typing.Type = None,
                  default_value: typing.Any = _not_found_,
                  getter=None,
                  **kwargs) -> _TObject:

        # 获得 type_hint
        if isinstance(key, str):
            attr = getattr(self.__class__, key, _not_found_)
            if isinstance(attr, sp_property):  # 若 key 是 sp_property
                if type_hint is None:
                    type_hint = attr.type_hint
                if getter is None:
                    getter = attr.getter
                if default_value is _not_found_:
                    default_value = attr.default_value
                kwargs = {**kwargs, **attr.kwargs}

            if type_hint is None:  # 作为一般属性，由type_hint决定类型
                type_hint = typing.get_type_hints(self.__class__).get(key, None)

        if type_hint is None:  # 由容器类型决定类型
            type_hint = typing.get_args(getattr(self, "__orig_class__", None))
            if len(type_hint) > 0:
                type_hint = type_hint[-1]

        orig_class = type_hint if inspect.isclass(type_hint) else typing.get_origin(type_hint)

        if value is _not_found_ and isinstance(key, (int, str)):
            # 如果 value 为 _not_found_, 则从 cache 中获取
            value = self._cache.get(key, _not_found_)

        if value is _not_found_ and key is not None:
            # 如果 value 为 _not_found_, 则从 self._entry 中获取
            value = self._entry.child(key, force=True)

        if getter is not None:  # 若定义 getter
            sig = inspect.signature(getter)
            if len(sig.parameters) == 1:
                value = getter(self)
            elif isinstance(value, Entry):
                value = getter(self, value.query(default_value=default_value), **kwargs)
            else:
                value = getter(self, value, **kwargs)

        if orig_class is None:  # 若 type_hint/orig_class 未定义，则由value决定类型
            if isinstance(value, Entry):
                value = value.query(default_value=default_value, **kwargs)
            elif value is _not_found_:
                value = default_value
        elif isinstance(value, orig_class):
            # 如果 value 符合 type_hint 则返回之
            return value  # type:ignore
        elif issubclass(orig_class, Node):  # 若 type_hint 为 Node
            value = type_hint(value,  parent=self, **kwargs)
        else:
            if isinstance(value, Entry):
                value = value.query(default_value=default_value, **kwargs)
            elif value is _not_found_:
                value = default_value

            if isinstance(value, orig_class):
                pass
            elif isinstance(type_hint, Container._PRIMARY_TYPE_):
                value = type_hint(value)
            elif dataclasses.is_dataclass(type_hint):
                value = as_dataclass(type_hint, value)
            else:
                value = type_hint(value, **kwargs)

                # raise TypeError(f"Illegal type hint {type_hint}")

        if not inspect.isclass(orig_class):
            pass
        elif not isinstance(value, orig_class):
            raise KeyError(f"Can not find {key}! type_hint={type_hint} value={type(value)}")
        elif isinstance(key, (str, int)):
            self._cache[key] = value

        return value

    def update_child(self,
                     value: typing.Optional[Entry] = None,
                     type_hint=None,
                     default_value: typing.Optional[typing.Any] = None,
                     getter:  typing.Optional[typing.Callable] = None,
                     in_place=True,
                     force=True,
                     *args, **kwargs) -> typing.Union[typing.Any, Node]:

        is_changed = True

        if value is None and key is not None:
            value = self._entry.child(key).query(default_value=_not_found_)
            is_changed = value is _not_found_

        is_valid = self.validate(value, type_hint) if value is not _not_found_ else False

        if not is_valid:
            if getter is not None:
                value = getter(self)
            elif value is _undefined_:
                value = default_value
            is_changed = True
            is_valid = self.validate(value, type_hint)

        if is_valid:
            obj = value
        elif type_hint is _undefined_:
            if isinstance(value, (collections.abc.Sequence, collections.abc.Mapping, Entry)) and not isinstance(value, str):
                obj = Node(value, *args, **kwargs)
            else:
                obj = value
            # obj = value if not isinstance(value, Entry) else value.dump()
        elif type_hint in Node._PRIMARY_TYPE_:  # (int, float, bool, str):
            if isinstance(value, Entry):
                value = value.query(default_value=_not_found_)
            elif hasattr(value, "__entry__"):
                value = value.__entry__.__value__
            if value is _undefined_ or isinstance(value, Entry):
                raise TypeError(value)
            elif type_hint is np.ndarray:
                obj = np.asarray(value)
            elif isinstance(value, tags):
                raise ValueError(f"Tags is not a value! key={key} tags={value}")
            else:
                try:
                    obj = type_hint(value)
                except TypeError as err:
                    raise TypeError(f"Can't convert value {value} to {type_hint}") from err

        elif dataclasses.is_dataclass(type_hint):
            if isinstance(value, collections.abc.Mapping):
                obj = type_hint(**{k: value.get(k, None) for k in type_hint.__dataclass_fields__})
            elif isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
                obj = type_hint(*value)
            else:
                obj = type_hint(value)
        elif inspect.isfunction(type_hint):
            obj = type_hint(value, *args,  **kwargs)
        elif inspect.isclass(type_hint):
            obj = type_hint.__new__(type_hint, value, *args, **kwargs)
            obj._parent = self
            obj.__init__(value, *args, **kwargs)
        elif typing.get_origin(type_hint) is not None:
            obj = type_hint(value, *args, **kwargs)
        else:
            raise NotImplementedError(type_hint)

        # elif hasattr(type_hint, '__origin__'):
            # if issubclass(type_hint.__origin__, Node):
            #     obj = type_hint(value, parent=parent, **kwargs)
            # else:
            #     obj = type_hint(value, **kwargs)
        # if inspect.isclass(type_hint):
        #     if issubclass(type_hint, Node):
        #         obj = type_hint(value, *args, parent=parent, **kwargs)
        # elif callable(type_hint):
        #     obj = type_hint(value, **kwargs)
        # else:
        #     if always_node:
        #         obj = Node(value, *args, parent=parent, **kwargs)
        #     logger.warning(f"Ignore type_hint={type(type_hint)}!")

        is_changed |= obj is not value

        ###################################################################

        if key is not _undefined_ and is_changed:
            if isinstance(obj, Entry) or isinstance(value, Entry):  # and self._entry._cache is value._cache:
                pass
            elif in_place and isinstance(key, (int, str)):
                self._entry.child(key).insert(obj)

        if isinstance(obj, Node):
            obj._parent = self

        return obj

        # if isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
        #     res = Node._SEQUENCE_TYPE_(value,  parent=parent, **kwargs)
        # elif isinstance(value, collections.abc.Mapping):
        #     res = Node._MAPPING_TYPE_(value,   parent=parent, **kwargs)
        # elif isinstance(value, Entry):
        #     if Node._LINK_TYPE_ is not None:
        #         res = Node._LINK_TYPE_(value,  parent=parent, **kwargs)
        #     else:
        #         res = Node(value,  parent=parent, **kwargs)
        # if isinstance(value, Node._PRIMARY_TYPE_) or isinstance(value, Node) or value in (None, _not_found_, _undefined_):
        #     return value

    # elif (isinstance(value, list) and all(filter(lambda d: isinstance(d, (int, float, np.ndarray)), value))):
    #     return value
    # elif inspect.isclass(self._new_child):
    #     if isinstance(value, self._new_child):
    #         return value
    #     elif issubclass(self._new_child, Node):
    #         return self._new_child(value, parent=parent, **kwargs)
    #     else:
    #         return self._new_child(value, **kwargs)
    # elif callable(self._new_child):
    #     return self._new_child(value, **kwargs)
    # elif isinstance(self._new_child, collections.abc.Mapping) and len(self._new_child) > 0:
    #     kwargs = collections.ChainMap(kwargs, self._new_child)
    # elif self._new_child is not None and not not self._new_child:
    #     logger.warning(f"Ignored!  { (self._new_child)}")

    # if isinstance(attribute, str) or attribute is None:
    #     attribute_type = self._attribute_type(attribute)
    # else:
    #     attribute_type = attribute

    # if inspect.isclass(attribute_type):
    #     if isinstance(value, attribute_type):
    #         res = value
    #     elif attribute_type in (int, float):
    #         res = attribute_type(value)
    #     elif attribute_type is np.ndarray:
    #         res = np.asarray(value)
    #     elif dataclasses.is_entryclass(attribute_type):
    #         if isinstance(value, collections.abc.Mapping):
    #             res = attribute_type(
    #                 **{k: value.get(k, None) for k in attribute_type.__entryclass_fields__})
    #         elif isinstance(value, collections.abc.Sequence):
    #             res = attribute_type(*value)
    #         else:
    #             res = attribute_type(value)
    #     elif issubclass(attribute_type, Node):
    #         res = attribute_type(value, parent=parent, **kwargs)
    #     else:
    #         res = attribute_type(value, **kwargs)
    # elif hasattr(attribute_type, '__origin__'):
    #     if issubclass(attribute_type.__origin__, Node):
    #         res = attribute_type(value, parent=parent, **kwargs)
    #     else:
    #         res = attribute_type(value, **kwargs)
    # elif callable(attribute_type):
    #     res = attribute_type(value, **kwargs)
    # elif attribute_type is not None:
    #     raise TypeError(attribute_type)

    # @property
    # def entry(self) -> Entry:
    #     return self._entry

    # def __ior__(self,  value: _T) -> _T:
    #     return self._entry.push({Entry.op_tag.update: value})

    # @property
    # def _is_list(self) -> bool:
    #     return False

    # @property
    # def _is_dict(self) -> bool:
    #     return False

    # @property
    # def is_valid(self) -> bool:
    #     return self._entry is not None

    # def flush(self):
    #     if self._entry.level == 0:
    #         return
    #     elif self._is_dict:
    #         self._entry.moveto([""])
    #     else:
    #         self._entry.moveto(None)

    # def clear(self):
    #     self._entry.push(Entry.op_tag.reset)

    # def remove(self, path: _TPath = None) -> bool:
    #     return self._entry.push(path, Entry.op_tag.remove)

    # def reset(self, cache=None, ** kwargs) -> None:
    #     if isinstance(cache, Entry):
    #         self._entry = cache
    #     elif cache is None:
    #         self._entry = None
    #     elif cache is not None:
    #         self._entry = Entry(cache)
    #     else:
    #         self._entry = Entry(kwargs)

    # def update(self, value: _T, **kwargs) -> _T:
    #     return self._entry.push([], {Entry.op_tag.update: value}, **kwargs)

    # def find(self, query: _TPath, **kwargs) -> _T:
    #     return self._entry.pull({Entry.op_tag.find: query},  **kwargs)

    # def try_insert(self, query: _TPath, value: _T, **kwargs) -> _T:
    #     return self._entry.push({Entry.op_tag.try_insert: {query: value}},  **kwargs)

    # def count(self, query: _TPath, **kwargs) -> int:
    #     return self._entry.pull({Entry.op_tag.count: query}, **kwargs)

    # # def dump(self) -> Union[Sequence, Mapping]:
    # #     return self._entry.pull(Entry.op_tag.dump)

    # def put(self, path: _TPath, value, *args, **kwargs) -> _T:
    #     return self._entry.put(path, value, *args, **kwargs)

    # def get(self, path: _TPath, *args, **kwargs) -> _T:
    #     return self._entry.get(path, *args, **kwargs)

    # def replace(self, path, value: _T, *args, **kwargs) -> _T:
    #     return self._entry.replace(path, value, *args, **kwargs)

    # def equal(self, path: _TPath, other) -> bool:
    #     return self._entry.pull(path, {Entry.op_tag.equal: other})
