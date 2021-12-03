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

    def __new__(cls, data=None, *args, **kwargs):
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

        return object.__new__(n_cls)

    @classmethod
    def create(cls, value: _T, /, parent=_undefined_,  **kwargs) -> Union[_T, _TNode]:

        if isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
            res = Node._SEQUENCE_TYPE_(value,  parent=parent, **kwargs)
        elif isinstance(value, collections.abc.Mapping):
            res = Node._MAPPING_TYPE_(value,   parent=parent, **kwargs)
        elif isinstance(value, Entry):
            if Node._CONTAINER_TYPE_ is not None:
                res = Node._CONTAINER_TYPE_(value,  parent=parent, **kwargs)
            else:
                res = Node(value,  parent=parent, **kwargs)
        # if isinstance(value, (Node._PRIMARY_TYPE_, Node)) or value in (None, _not_found_, _undefined_):
        #     res = value
        else:
            res = value

        return res

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
