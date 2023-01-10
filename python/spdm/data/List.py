import collections
import collections.abc
import typing

from ..common.tags import _not_found_, _undefined_
from ..util.logger import logger
from ..util.utilities import serialize
from .Container import Container
from .Entry import Entry, EntryCombine, as_entry
from .Node import Node

_TList = typing.TypeVar('_TList', bound='List')
_TObject = typing.TypeVar("_TObject")
_T = typing.TypeVar("_T")


class List(Container[_TObject], typing.Sequence[_TObject]):

    def __init__(self,  *args,  **kwargs) -> None:
        if len(args) == 1:
            if args[0] in (_undefined_, _not_found_, None):
                args = []
            elif isinstance(args[0], Entry):
                args = args[0]
            elif isinstance(args[0], typing.Sequence) and not isinstance(args[0], str):
                args = list(args[0])
            else:
                args = list(args)
        else:
            args = list(args)

        super().__init__(args)

    @property
    def _is_list(self) -> bool:
        return True

    def __serialize__(self) -> list:
        return [serialize(v) for v in self._entry.first_child()]

    def __len__(self) -> int:
        return self._entry.count()

    def __getitem__(self, key) -> _TObject:
        return self._post_process(self._entry.child(key), key=key)

    def __setitem__(self, key, value: _T) -> None:
        return self._entry.child(key).push(value)

    def __delitem__(self,  key) -> None:
        return self._entry.child(key).erase()

    def __iter__(self) -> typing.Iterator[_TObject]:
        for idx, v in enumerate(self._entry.first_child()):
            yield self._post_process(v, key=idx)

    def __iadd__(self, value) -> _TList:
        self._entry.push(value, append=True)
        return self

    def append(self, value) -> _TList:
        self._entry.push([value], append=True)
        return self

    def update(self, d, predication=_undefined_, only_first=False) -> int:
        return self._entry.push([], self._pre_process(d), predication=predication, only_first=only_first)

    def sort(self) -> None:
        self._entry.sort()

    def combine(self, default=_undefined_, reducer=_undefined_, partition=_undefined_, **kwargs) -> _T:
        res = EntryCombine([m for m in self.__iter__()], reducer=reducer, partition=partition,
                           default=default if default is not _undefined_ else self._combiner,
                           **kwargs)
        return self._post_process(res, key=_undefined_)

    def find(self, predication,  only_first=True) -> _T:
        return self._post_process(self._entry.pull(predication=predication, only_first=only_first))

    def _post_process(self, value: _T, key, *args,  ** kwargs) -> typing.Union[_T, Node]:
        obj = super()._post_process(value, key, *args,  ** kwargs)
        if isinstance(obj, Node):
            obj._parent = self._parent
        return obj

    class ListAsEntry(Entry):
        def __init__(self, cache, **kwargs):
            super().__init__(cache, **kwargs)

        def pull(self, default=...) -> typing.Any:
            if len(self._path) > 0 and not isinstance(self._path[0], int):
                obj = self._cache.__getitem__(self._path[0])
                if len(self._path) == 1:
                    return obj
                else:
                    return as_entry(obj).child(self._path[1:]).pull(default)
            else:
                return self._cache._entry.child(self._path).pull(default)

        def push(self, value: typing.Any, **kwargs) -> None:
            self._cache._entry.child(self._path).push(value, **kwargs)

        def erase(self) -> bool:
            if len(self._path) == 1 and not isinstance(self._path[0], str) and self._path[0] in self._cache._properties:
                delattr(self._cache, self._path[0])
            return self._cache._entry.child(self._path).erase()

    def __entry__(self) -> Entry:
        return List.ListAsEntry(self)
