import collections
import collections.abc
import dataclasses
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

    def __new__(cls,  *args, **kwargs):
        if cls is not Node:
            obj = object.__new__(cls)
        elif len(args) == 1:
            if isinstance(args[0], collections.abc.Sequence) and not isinstance(args[0], str):
                obj = object.__new__(Node._SEQUENCE_TYPE_)
            elif isinstance(args[0], collections.abc.Mapping):
                obj = object.__new__(Node._MAPPING_TYPE_)
            elif isinstance(args[0], Entry) and Node._LINK_TYPE_ is not None:
                obj = object.__new__(Node._LINK_TYPE_)
            else:
                obj = object.__new__(cls)
        elif len(args) > 1:
            obj = object.__new__(Node._SEQUENCE_TYPE_)
        elif len(kwargs) > 0:
            obj = object.__new__(Node._MAPPING_TYPE_)
        else:
            obj = object.__new__(cls)

        return obj

    def __init__(self, data=None) -> None:
        super().__init__()
        self._entry = data if isinstance(data, Entry) else Entry(data)
        self._nid = None
        self._parent = None

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

    def _post_process(self, value: _T, path=_undefined_, *args,   ** kwargs) -> Union[_T, _TNode]:
        return self.update_child(value, *args, path=path,  **kwargs)

    def update_child(self, value: _T, path=_undefined_, *args,   ** kwargs) -> Union[_T, _TNode]:
        if value is _undefined_ and path is not _undefined_:
            value = self._entry.child(path).pull(_undefined_)

        return self.create_child(value, path, *args, **kwargs)

    def create_child(self, value: _T,  path=_undefined_,
                     *args,
                     parent=_undefined_,
                     type_hint=_undefined_,
                     always_node=False, **kwargs) -> Union[_T, _TNode]:

        if not always_node and isinstance(value, Node._PRIMARY_TYPE_):
            return value

        if parent is _undefined_:
            parent = self

        obj = _undefined_

        # if type_hint is _undefined_:
        #     type_hint = Node

        if type_hint is _undefined_:
            obj = value
        elif type_hint in (int, float, bool):
            obj = type_hint(value)
        elif type_hint is np.ndarray:
            obj = np.asarray(value)
        elif dataclasses.is_dataclass(type_hint):
            if isinstance(value, collections.abc.Mapping):
                obj = type_hint(**{k: value.get(k, None) for k in type_hint.__dataclass_fields__})
            elif isinstance(value, collections.abc.Sequence):
                obj = type_hint(*value)
            else:
                obj = type_hint(value)
        elif inspect.isfunction(type_hint):
            obj = type_hint(value, *args,  **kwargs)
        elif inspect.isclass(type_hint):
            if isinstance(value, type_hint):
                obj = value
            elif issubclass(type_hint, Node):
                obj = type_hint(value, **kwargs)
        elif hasattr(type_hint, "__origin__"):
            obj = type_hint(value, **kwargs)
        else:
            obj = type_hint(value, **kwargs)

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

        if obj is _undefined_:
            obj = value
        elif path is not _undefined_ and not isinstance(value, Entry):
            self._entry.child(path).push(obj)

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
