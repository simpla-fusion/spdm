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
from .Entry import (_DICT_TYPE_, _LIST_TYPE_, Entry, EntryCombiner, _next_,
                    _TPath)
from .Node import Node

_TList = TypeVar('_TList', bound='List')
_TObject = TypeVar("_TObject")
_T = TypeVar("_T")


class List(Container, Sequence[_TObject]):

    def __init__(self, cache: Union[Sequence, Entry] = None, /,   **kwargs) -> None:
        super().__init__(cache if cache is not None else _LIST_TYPE_(),  **kwargs)

    def _serialize(self) -> Sequence:
        return [serialize(v) for v in self.__iter__()]

    @property
    def _is_list(self) -> bool:
        return True

    def __len__(self) -> int:
        return super().__len__()

    def __setitem__(self, path: _TPath, v: _T) -> None:
        return super().__setitem__(path,  v)

    def __getitem__(self, path: _TPath) -> _TObject:
        return super().__getitem__(path)

    def __delitem__(self, path: _TPath) -> None:
        return super().__delitem__(path)

    def __iter__(self) -> Iterator[_TObject]:
        yield from super().__iter__()

    def __iadd__(self, other):
        self._entry.put(_next_, other)
        return self

    def sort(self):
        if hasattr(self._entry.__class__, "sort"):
            self._entry.sort()
        else:
            raise NotImplementedError()

    def filter(self, predication) -> _TList:
        return self.__class__(self._entry.filter(predication))

    def combine(self, default_value=None, predication=_undefined_, reducer=_undefined_, partition=_undefined_) -> _TObject:
        return self._post_process(EntryCombiner(self._entry, default_value=default_value,  reducer=reducer, partition=partition))

    def reset(self, value=None):
        if isinstance(value, (collections.abc.Sequence)):
            super().reset(value)
        else:
            self._combine = value
            super().reset()

    def find(self, predication,  only_first=True) -> _TObject:
        return self._post_process(self._entry.pull(predication=predication, only_first=only_first))

    def update(self, d, predication=_undefined_, only_first=False) -> int:
        return self._entry.push([], self._pre_process(d), predication=predication, only_first=only_first)
