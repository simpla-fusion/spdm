import collections
import collections.abc
from logging import log
from typing import (Any, Callable, Generic, Iterator, Mapping, MutableMapping,
                    MutableSequence, Optional, Sequence, Tuple, Type, TypeVar,
                    Union, final, get_args)

import numpy as np

from ..common.logger import logger
from ..common.tags import _not_found_, _undefined_
from ..util.utilities import serialize
from .Container import Container
from .Entry import Entry, EntryChain, EntryCombiner, _TPath
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

    def __setitem__(self, idx, v: _T) -> None:
        return self._entry.child(idx).push(v)

    def __getitem__(self, idx) -> _TObject:
        return self._post_process(self._entry.child(idx), key=idx)

    def __delitem__(self, idx) -> None:
        self._entry.child(idx).erase()

    def __iter__(self) -> Iterator[_TObject]:
        for idx, v in enumerate(self._entry.first_child()):
            yield self._post_process(v, key=idx)

    def __iadd__(self, other) -> _TList:
        self._entry.push(other, append=True)
        return self

    def sort(self) -> None:
        self._entry.sort()

    def combine(self, default=_undefined_, reducer=_undefined_, partition=_undefined_) -> _TObject:
        e = EntryCombiner([m for m in self.__iter__()], reducer=reducer, partition=partition)
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
