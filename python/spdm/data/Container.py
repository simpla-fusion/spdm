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

_TKey = typing.TypeVar("_TKey")
_TObject = typing.TypeVar("_TObject")
_T = typing.TypeVar("_T")


class Container(Node, typing.Generic[_TObject]):
    r"""
       Container Node
    """

    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @cached_property
    def _child_type(self):

        child_type = _undefined_
        #  @ref: https://stackoverflow.com/questions/48572831/how-to-access-the-type-arguments-of-typing-generic?noredirect=1
        orig_class = getattr(self, "__orig_class__", _not_found_)
        if orig_class is not _not_found_:
            child_type = typing.get_args(self.__orig_class__)
            if len(child_type) > 0 and inspect.isclass(child_type[0]):
                child_type = child_type[0]
        return child_type

    def update_child(self, key: _TKey, value: _T = _undefined_,   type_hint=_undefined_, *args, **kwargs) -> typing.Union[_T, Node]:
        return super().update_child(key,
                                    value,
                                    type_hint=type_hint if type_hint is not _undefined_ else self._child_type,
                                    *args, **kwargs)

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
    # elif self._new_child is not _undefined_ and not not self._new_child:
    #     logger.warning(f"Ignored!  { (self._new_child)}")

    # if isinstance(attribute, str) or attribute is _undefined_:
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
    # elif attribute_type is not _undefined_:
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

    # def reset(self, cache=_undefined_, ** kwargs) -> None:
    #     if isinstance(cache, Entry):
    #         self._entry = cache
    #     elif cache is None:
    #         self._entry = None
    #     elif cache is not _undefined_:
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
