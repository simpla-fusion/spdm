import collections
import collections.abc
import dataclasses
import inspect
from functools import cached_property
from typing import Any, Generic, Iterator, TypeVar, Union, final, get_args, Mapping

import numpy as np

from ..common.logger import logger
from ..common.tags import _not_found_, _undefined_
from ..util.utilities import serialize
from .Entry import (Entry,   _next_, _TPath)
from .Node import Node
from .Path import Path

_TObject = TypeVar("_TObject")
_TLink = TypeVar("_TLink", bound="Link")
_T = TypeVar("_T")


class Link(Node):
    r"""
       Container Node
    """

    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        annotation = [f"{k}='{v}'" for k, v in self.annotation.items() if v is not None]
        return f"<{getattr(self,'__orig_class__',self.__class__.__name__)} {' '.join(annotation)}/>"

    def __serialize__(self) -> Any:
        return serialize(self._entry.dump())

    def _duplicate(self, *args, parent=None, **kwargs) -> _TLink:
        return self.__class__(self._entry, *args, parent=parent if parent is not None else self._parent,  **kwargs)

    def __setitem__(self, key: Any, value: _T) -> _T:
        return self._entry.child(key).push(self._pre_process(value))

    def __getitem__(self, key: Any) -> Any:
        return self._post_process(self._entry.child(key), key=key)

    def __delitem__(self, key: Any) -> bool:
        return self._entry.child(key).erase()

    def __contains__(self, key: Any) -> bool:
        return self._entry.child(key).exists()

    def __eq__(self, other) -> bool:
        return self._entry.equal(other)

    def __len__(self) -> int:
        return self._entry.count()

    def __iter__(self) -> Iterator[_T]:
        for idx, obj in enumerate(self._entry):
            yield self._post_process(obj, key=[idx])

    def append(self, value) -> _TLink:
        self._entry.push([value], extend=True)
        return self

    def extend(self, value) -> _TLink:
        self._entry.push(value, extend=True)
        return self

    def __ior__(self, obj) -> _TLink:
        self._entry.push(obj, update=True)
        return self
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

    # def find(self, query: _TPath, **kwargs) -> _TObject:
    #     return self._entry.pull({Entry.op_tag.find: query},  **kwargs)

    # def try_insert(self, query: _TPath, value: _T, **kwargs) -> _T:
    #     return self._entry.push({Entry.op_tag.try_insert: {query: value}},  **kwargs)

    # def count(self, query: _TPath, **kwargs) -> int:
    #     return self._entry.pull({Entry.op_tag.count: query}, **kwargs)

    # # def dump(self) -> Union[Sequence, Mapping]:
    # #     return self._entry.pull(Entry.op_tag.dump)

    # def put(self, path: _TPath, value, *args, **kwargs) -> _TObject:
    #     return self._entry.put(path, value, *args, **kwargs)

    # def get(self, path: _TPath, *args, **kwargs) -> _TObject:
    #     return self._entry.get(path, *args, **kwargs)

    # def replace(self, path, value: _T, *args, **kwargs) -> _T:
    #     return self._entry.replace(path, value, *args, **kwargs)

    # def equal(self, path: _TPath, other) -> bool:
    #     return self._entry.pull(path, {Entry.op_tag.equal: other})
