import collections
import collections.abc
from typing import (Any, Generic, Iterator, Mapping, TypeVar, Union, final,
                    get_args)

import numpy as np

from ..common.logger import logger
from ..common.SpObject import SpObject
from ..common.tags import _not_found_, _undefined_
from ..util.utilities import serialize
from .Entry import Entry

_T = TypeVar("_T")
_TObject = TypeVar("_TObject")
_TNode = TypeVar("_TNode", bound="Node")


class Node(SpObject):
    __slots__ = "_entry"

    _PRIMARY_TYPE_ = (bool, int, float, str, np.ndarray)
    _MAPPING_TYPE_ = dict
    _SEQUENCE_TYPE_ = list
    _CONTAINER_TYPE_ = None

    def __new__(cls, data, *args, **kwargs):
        if cls is not Node:
            n_cls = cls
        elif isinstance(data, collections.abc.Sequence):
            n_cls = Node._SEQUENCE_TYPE_
        elif isinstance(data, collections.abc.Mapping):
            n_cls = Node._MAPPING_TYPE_
        elif isinstance(data, Entry):
            n_cls = Node._CONTAINER_TYPE_
        else:
            n_cls = cls

        return object.__new__(cls)

    def __init__(self, data, *args, parent=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._parent = parent
        self._entry = data if isinstance(data, Entry) else Entry(data)

    def dump(self):
        return self._serialize()

    def _serialize(self):
        return serialize(self._entry)

    def _pre_process(self, value: _T, *args, **kwargs) -> _T:
        return value

    def _post_process(self, value: _T, *args,   **kwargs) -> Union[_T, _TNode]:
        return Node._convert(value, self, *args,  **kwargs)

    @staticmethod
    def _convert(value: _T,    parent, *args, **kwargs) -> Union[_T, _TNode]:

        if isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
            res = Node._SEQUENCE_TYPE_(value, *args, parent=parent, **kwargs)
        elif isinstance(value, collections.abc.Mapping):
            res = Node._MAPPING_TYPE_(value, *args, parent=parent, **kwargs)
        elif isinstance(value, Entry):
            if Node._CONTAINER_TYPE_ is not None:
                res = Node._CONTAINER_TYPE_(value, *args, parent=parent, **kwargs)
            else:
                res = Node(value, *args, parent=parent, **kwargs)
        # if isinstance(value, (Node._PRIMARY_TYPE_, Node)) or value in (None, _not_found_, _undefined_):
        #     res = value
        else:
            res = value

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

        return res

    def reset(self):
        self._entry = None
