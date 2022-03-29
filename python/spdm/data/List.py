import collections
import collections.abc
from logging import log
from typing import (Any, Callable, Generic, Iterator, Mapping, MutableMapping,
                    MutableSequence, Optional, Sequence, Tuple, Type, TypeVar,
                    Union, final, get_args)

import numpy as np

from spdm.logger import logger
from spdm.tags import _not_found_, _undefined_
from ..util.utilities import serialize
from .Container import Container
from .Entry import Entry, EntryChain, EntryCombine, as_entry
from .Node import Node

_TList = TypeVar('_TList', bound='List')
_TObject = TypeVar("_TObject")
_T = TypeVar("_T")


class List(Container[_TObject], Sequence[_TObject]):

    def __init__(self,  *args) -> None:
        if len(args) != 1:
            args = list(args)
        elif isinstance(args[0], Entry):
            args = args[0]
        elif isinstance(args[0], Sequence) and not isinstance(args[0], str):
            args = list(args[0])
        elif args[0] is _undefined_ or args[0] is _not_found_:
            args = []
        else:
            args = list(args)

        super().__init__(args)

    def __serialize__(self) -> Sequence:
        return [serialize(v) for v in self.__iter__()]

    @property
    def _is_list(self) -> bool:
        return True

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, key) -> _TObject:
        return super().__getitem__(key)

    def __setitem__(self, key, value: _T) -> None:
        super().__setitem__(key, value)

    def __delitem__(self,  key) -> None:
        super().__delitem__(key)

    def __iter__(self) -> Iterator[_TObject]:
        for idx, v in enumerate(self._entry.first_child()):
            yield self._post_process(v, key=idx)

    def __iadd__(self, other) -> _TList:
        self._entry.push(other, append=True)
        return self

    class ListAsEntry(Entry):
        def __init__(self, cache, **kwargs):
            super().__init__(cache, **kwargs)

        def pull(self, default=...) -> Any:
            if len(self._path) > 0 and not isinstance(self._path[0], int):
                obj = self._cache.__getitem__(self._path[0])
                if len(self._path) == 1:
                    return obj
                else:
                    return as_entry(obj).child(self._path[1:]).pull(default)
            else:
                return self._cache._entry.child(self._path).pull(default)

        def push(self, value: Any, **kwargs) -> None:
            self._cache._entry.child(self._path).push(value, **kwargs)

        def erase(self) -> bool:
            if len(self._path) == 1 and not isinstance(self._path[0], str) and self._path[0] in self._cache._properties:
                delattr(self._cache, self._path[0])
            return self._cache._entry.child(self._path).erase()

    def __entry__(self) -> Entry:
        return List.ListAsEntry(self)

    def sort(self) -> None:
        self._entry.sort()

    def combine(self, default=_undefined_, reducer=_undefined_, partition=_undefined_) -> _TObject:
        e = EntryCombine([m for m in self.__iter__()], reducer=reducer, partition=partition)
        default = default if default is not _undefined_ else {}

        return self._post_process(EntryChain([default, e]), key=_undefined_)

    def find(self, predication,  only_first=True) -> _TObject:
        return self._post_process(self._entry.pull(predication=predication, only_first=only_first))

    def update(self, d, predication=_undefined_, only_first=False) -> int:
        return self._entry.push([], self._pre_process(d), predication=predication, only_first=only_first)

    def _post_process(self, value: _T, key, *args,  ** kwargs) -> Union[_T, Node]:
        obj = super()._post_process(value, key, *args,  ** kwargs)
        if isinstance(obj, Node):
            obj._parent = self._parent
        return obj
