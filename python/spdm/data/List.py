from __future__ import annotations

import collections.abc
import typing

from ..utils.misc import serialize
from ..utils.tags import _not_found_, _undefined_
from .Container import Container
from .Entry import as_entry
from .Node import Node
from .Path import Path

_TObject = typing.TypeVar("_TObject")


class List(Container[_TObject], typing.Sequence[_TObject]):

    def __init__(self, *args, ** kwargs):
        super().__init__(*args, **kwargs)

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

    def __iter__(self) -> typing.Generator[_TObject, None, None]:
        for idx, v in enumerate(self._entry.child(slice(None)).find()):
            yield self._as_child(idx, v)

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

    def _as_child(self, key: int | slice,  value=_not_found_, *args, default_value=_not_found_, **kwargs) -> _TObject:

        if default_value is _not_found_:
            # 如果没有指定 default_value，则使用 self._default_value
            default_value = self._default_value

        if isinstance(key, int):
            n_value = super()._as_child(key, value, *args, default_value=default_value,  **kwargs)
            if isinstance(n_value, Node) and n_value._parent is self:
                n_value._parent = self._parent
        elif isinstance(key, slice):
            if key.start is None or key.stop is None or key.step is None:
                raise ValueError(f"slice must be a complete slice {key}")
            if isinstance(value, collections.abc.Sequence):
                if len(value) == (key.stop-key.start)/key.step:
                    raise ValueError(f"value must be a sequence with length {(key.stop-key.start)/key.step} {value}")
                n_value = [self._as_child(idx, value[idx], *args, default_value=default_value, **kwargs)
                           for idx in range(key.start, key.stop, key.step)]
            elif isinstance(value, collections.abc.Generator):
                n_value = []
                for idx in range(key.start, key.stop, key.step):
                    n_value.append(self._as_child(idx, next(value), *args, default_value=default_value, **kwargs))
            else:
                raise TypeError(f"key must be int or slice, not {type(key)}")

        return n_value

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

    def find(self, predication, **kwargs) -> typing.Generator[typing.Any, None, None]:
        yield from self._entry.child(predication).find(**kwargs)


Node._SEQUENCE_TYPE_ = List
