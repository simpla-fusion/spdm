import collections
import collections.abc
import dataclasses
import inspect
import typing
from functools import cached_property

import numpy as np

from ..common.tags import _not_found_, _undefined_
from ..util.logger import logger
from .Node import Node
from .Function import Function
from .Entry import Entry, as_entry
from .Path import Path
_TKey = typing.TypeVar("_TKey")
_TObject = typing.TypeVar("_TObject")
_T = typing.TypeVar("_T")


class Container(Node, typing.Generic[_TKey, _TObject]):
    r"""
       Container Node
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}

    def __serialize__(self) -> dict:
        return self._entry.__serialize__()

    def __setitem__(self, *args) -> typing.Any:
        path = args[:-1]
        value = args[-1]
        if isinstance(value, Node):
            raise NotImplementedError()
        else:
            self._entry.child(path).insert(value)
        return value

    def __getitem__(self, *args) -> typing.Any:
        path = Path(args)
        obj = self
        for idx, key in enumerate(path[:]):
            if isinstance(obj, Container):
                obj = obj._as_child(key, as_attribute=False)
                continue
            else:
                obj = as_entry(obj).child(path[idx:], force=True).query()
                break

        return obj

    def __delitem__(self, key) -> bool:
        return self._entry.child(key).remove() > 0

    def __contains__(self, key) -> bool:
        return self._entry.child(key).exists

    def __eq__(self, other) -> bool:
        return self._entry.equal(other)

    def __len__(self) -> int:
        return self._entry.count

    def _as_child(self, key: _TKey, type_hint: typing.Type = None,
                  default_value: typing.Any = _not_found_,
                  getter=None, **kwargs) -> typing.Any:

        obj = self._cache.get(key, None) or self._entry.child(key, force=True)

        type_hint = type_hint \
            or typing.get_type_hints(self.__class__).get(str(key), None) \
            or self._child_type

        if type_hint is None:
            if not isinstance(obj, Node._PRIMARY_TYPE_):
                obj = Node(as_entry(obj), parent=self)
        elif not callable(type_hint):  # inspect.isclass(type_hint)
            raise TypeError(type_hint)
        elif inspect.isclass(type_hint) and isinstance(obj, type_hint):
            pass
        elif issubclass(type_hint, Node):
            obj = type_hint(as_entry(obj), parent=self)
        elif type_hint in Node._PRIMARY_TYPE_ or issubclass(type_hint, Function):  # (int, float, bool, str):
            obj = as_entry(obj).query(default_value=_not_found_)
            if obj is not _not_found_:
                pass
            elif getter is not None:
                obj = getter(self)
            else:
                obj = default_value

        if obj is not _not_found_:
            self._cache[key] = obj

        return obj

    @cached_property
    def _child_type(self) -> typing.Type:
        child_type = None
        #  @ref: https://stackoverflow.com/questions/48572831/how-to-access-the-type-arguments-of-typing-generic?noredirect=1
        orig_class = getattr(self, "__orig_class__", None)
        if orig_class is not None:
            child_type = typing.get_args(self.__orig_class__)
            if len(child_type) > 0 and inspect.isclass(child_type[-1]):
                child_type = child_type[-1]

        return child_type

    # def update_child(self, key: typing.Any, value: _T = None,   type_hint=None, *args, **kwargs) -> typing.Union[_T, _TObject]:
    #     return super().update_child(key,
    #                                 value,
    #                                 type_hint=type_hint if type_hint is not None else self._child_type,
    #                                 *args, **kwargs)

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
