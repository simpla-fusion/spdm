from __future__ import annotations

import collections
import collections.abc
import typing

from ..common.tags import _not_found_, _undefined_
from ..util.logger import logger
from ..util.misc import serialize
from .Container import Container
from .Entry import Entry, EntryCombine, as_entry
from .Node import Node
from .Path import Path

_TObject = typing.TypeVar("_TObject")


class List(Container[_TObject], typing.Sequence[_TObject]):

    def __init__(self, *args, cache=None, **kwargs):
        super().__init__(*args, cache=[] if cache is None else cache, **kwargs)

    @property
    def _is_list(self) -> bool:
        return True

    def __serialize__(self) -> list:
        return [serialize(v) for v in self._entry.first_child()]

    def __len__(self) -> int:
        return self._entry.count

    # def __setitem__(self, key: int, value: typing.Any):
    #     return self._entry.child(key).insert(value)

    # def __delitem__(self,  key):
    #     return self._entry.child(key).remove()

    def __getitem__(self, path) -> _TObject:
        return super().__getitem__(path)

    def _as_child(self, *args, **kwargs) -> _TObject:
        obj = super()._as_child(*args, **kwargs)
        if isinstance(obj, Node) and obj._parent is self:
            obj._parent = self._parent
        return obj

    def __iter__(self) -> typing.Generator[_TObject, None, None]:
        for idx, v in enumerate(self._entry.child(slice(None)).find()):
            yield self._as_child(idx, v)

    def __iadd__(self, value) -> List:
        self._entry.update({Path.tags.append: value})
        return self

    def append(self, value) -> List:
        self._entry.update({Path.tags.append:  [value]})
        return self

    def update(self, d, predication=_undefined_, **kwargs) -> int:
        return self._entry.child(predication).update(d, **kwargs)

    def sort(self) -> None:
        self._entry.update(Path.tags.sort)

    def flash(self):
        for idx, item in enumerate(self._entry.child(slice(None)).find()):
            self._as_child(idx, item)
        return self

    def combine(self, selector=None,   **kwargs) -> _TObject:

        self.flash()
        if selector == None:
            return self._as_child(None, as_entry(self._cache).combine(**kwargs))
        else:
            return self._as_child(None, as_entry(self._cache).child(selector).combine(**kwargs))

    def find(self, predication, **kwargs) -> typing.Generator[typing.Any, None, None]:
        yield from self._entry.child(predication).find(**kwargs)


Node._SEQUENCE_TYPE_ = List
