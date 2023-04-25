from __future__ import annotations

import collections
import collections.abc
import dataclasses
import inspect
import typing

import numpy as np

from ..utils.tags import _not_found_, _undefined_, tags
from .Entry import Entry, as_entry


class Node(object):
    """

    """

    _PRIMARY_TYPE_ = (bool, int, float, str, np.ndarray)
    _MAPPING_TYPE_ = dict
    _SEQUENCE_TYPE_ = list

    def __new__(cls,  *args, **kwargs):
        if cls is not Node:
            return object.__new__(cls)
        if len(args) == 0:
            n_cls = Node
        elif hasattr(args[0], "__as__entry__"):
            if args[0].__as__entry__().is_sequence:
                n_cls = Node._SEQUENCE_TYPE_
            elif args[0].__as__entry__().is_mapping:
                n_cls = Node._MAPPING_TYPE_
            else:
                n_cls = cls
        elif isinstance(args[0], collections.abc.Sequence) and not isinstance(d, str):
            n_cls = Node._SEQUENCE_TYPE_
        elif isinstance(args[0], collections.abc.Mapping):
            n_cls = Node._MAPPING_TYPE_
        else:
            n_cls = cls

        if n_cls in (dict, list):
            return n_cls.__new__(n_cls)
        else:
            return object.__new__(n_cls)

    def __init__(self, entry=None, *args, parent=None, **kwargs) -> None:
        super().__init__()
        self._entry = as_entry(entry)
        self._parent = parent

    def duplicate(self) -> Node:
        other: Node = Node.__new__(self.__class__)
        other._entry = self._entry.duplicate()
        other._parent = self._parent
        return other

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} />"

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}>{self._entry.dump()}</{self.__class__.__name__}>"

    def flash(self):
        self._entry = Entry(self._entry.dump())

    @property
    def annotation(self) -> dict:
        return {"id": self.nid,   "type":  self._entry.__class__.__name__}

    @property
    def nid(self) -> str:
        return self._nid

    def __entry__(self) -> Entry:
        return self._entry

    
    def __value__(self) -> typing.Any:
        return self._entry.__value__()

    def reset(self):
        self._entry.reset()

    def dump(self):
        return self.__serialize__()

    def __serialize__(self):
        return self._entry.dump()

    def validate(self, value, type_hint) -> bool:
        if value is _undefined_ or type_hint is _undefined_:
            return False
        else:
            v_orig_class = getattr(value, "__orig_class__", value.__class__)

            if inspect.isclass(type_hint) and inspect.isclass(v_orig_class) and issubclass(v_orig_class, type_hint):
                res = True
            elif typing.get_origin(type_hint) is not None \
                and typing.get_origin(v_orig_class) is typing.get_origin(type_hint) \
                    and typing.get_args(v_orig_class) == typing.get_args(type_hint):
                res = True
            else:
                res = False
        return res
