import collections
import collections.abc
import inspect
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
    _LINK_TYPE_ = None

    def __new__(cls, value=None, *args, **kwargs):
        if cls is not Node:
            obj = object.__new__(cls)
        elif isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
            obj = object.__new__(Node._SEQUENCE_TYPE_)
        elif isinstance(value, collections.abc.Mapping):
            obj = object.__new__(Node._MAPPING_TYPE_)
        elif isinstance(value, Entry) and Node._LINK_TYPE_ is not None:
            obj = object.__new__(Node._LINK_TYPE_)
        else:
            obj = object.__new__(cls)

        return obj

    def __init__(self, data, *args, parent=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._parent = parent
        self._entry = data if isinstance(data, Entry) else Entry(data)
        self._nid = None

    @property
    def annotation(self) -> dict:
        return {"id": self.nid,   "type":  self._entry.__class__.__name__}

    @property
    def nid(self) -> str:
        return self._nid

    @property
    def entry(self) -> Entry:
        return self._entry

    def reset(self):
        self._entry = None

    def dump(self):
        return self._entry.dump()

    def __serialize__(self):
        return self._entry.dump()

    def _pre_process(self, value: _T, *args, **kwargs) -> _T:
        return value

    def _post_process(self, value: _T, key=_undefined_, *args,   ** kwargs) -> Union[_T, _TNode]:
        return self.update_child(value, key=key, *args, **kwargs)

    def update_child(self, value: _T, key=_undefined_, *args,   ** kwargs) -> Union[_T, _TNode]:
        if value is _undefined_ and key is not _undefined_:
            value = self._entry.child(key).pull(_undefined_)

        return self.create_child(value, key, *args, **kwargs)

    def create_child(self, value: _T,  key=_undefined_,
                     *args,
                     parent=_undefined_,
                     type_hint=_undefined_,
                     always_node=False, **kwargs) -> Union[_T, _TNode]:

        if not always_node and isinstance(value, Node._PRIMARY_TYPE_):
            return value

        obj = _undefined_

        if parent is _undefined_:
            parent = self

        if type_hint is _undefined_:
            type_hint = Node

        if inspect.isclass(type_hint):
            if issubclass(type_hint, Node):
                obj = type_hint(value, *args, parent=parent, **kwargs)
            else:
                obj = type_hint(value)
        elif callable(type_hint):
            obj = type_hint(value, **kwargs)
        else:
            if always_node:
                obj = Node(value, *args, parent=parent, **kwargs)
            logger.warning(f"Ignore type_hint={type(type_hint)}!")

        if obj is _undefined_:
            obj = value
        elif key is not _undefined_:
            self._entry.child(key).push(obj)

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
