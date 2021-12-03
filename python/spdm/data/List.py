import collections
import collections.abc
from typing import (Any, Callable, Generic, Iterator, Mapping, MutableMapping,
                    MutableSequence, Optional, Sequence, Tuple, Type, TypeVar,
                    Union, final, get_args)

import numpy as np

from ..common.logger import logger
from ..common.tags import _not_found_, _undefined_
from ..util.utilities import serialize
from .Container import Container
from .Entry import (_LIST_TYPE_, Entry, EntryCombiner, _next_,  _TPath)
from .Node import Node

_TList = TypeVar('_TList', bound='List')
_TObject = TypeVar("_TObject")
_T = TypeVar("_T")


class List(Container[_TObject], Sequence[_TObject]):

    def __init__(self, data: Sequence = None, /,   **kwargs) -> None:
        super().__init__(data if data is not None else list(), **kwargs)

    def _serialize(self) -> Sequence:
        return [serialize(v) for v in self.__iter__()]

    @property
    def _is_list(self) -> bool:
        return True

    def __len__(self) -> int:
        return self._entry.count()

    def __setitem__(self, idx, v: _T) -> None:
        return self._entry.child(idx).set_value(v)

    def __getitem__(self, idx) -> _TObject:
        return self._post_process(self._entry.child(idx).get_value(), path=idx)

    def __delitem__(self, idx) -> None:
        self._entry.child(idx).remove()

    def __iter__(self) -> Iterator:
        for idx, v in enumerate(self._entry.first_child()):
            yield self._post_process(self._entry, path=idx)

    def __iadd__(self, other) -> _TList:
        self._entry.extend(other)
        return self
    

    def sort(self) -> None:
        self._entry.sort()

    def filter(self, predication) -> _TList:
        return self.__class__(self._entry.filter(predication))

    def combine(self, default_value=None, predication=_undefined_, reducer=_undefined_, partition=_undefined_) -> _TObject:
        return self._post_process(EntryCombiner(self._entry, default_value=default_value,  reducer=reducer, partition=partition))

    def find(self, predication,  only_first=True) -> _TObject:
        return self._post_process(self._entry.pull(predication=predication, only_first=only_first))

    def update(self, d, predication=_undefined_, only_first=False) -> int:
        return self._entry.push([], self._pre_process(d), predication=predication, only_first=only_first)


Node._SEQUENCE_TYPE_ = List[Node]
