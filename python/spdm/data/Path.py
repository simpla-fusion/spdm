import collections.abc
from copy import deepcopy
from typing import TypeVar

_TPath = TypeVar("_TPath", bound="Path")


class Path(object):
    SEPERATOR = '.'

    def __init__(self, d=None, *args, **kwargs):
        self._items = list(d) if d is not None else []

    def __repr__(self):
        return Path.SEPERATOR.join([str(d) for d in self._items])

    def duplicate(self) -> _TPath:
        return Path(self._items)

    def append(self, *args) -> _TPath:
        for item in args:
            if isinstance(item, str):
                self._items.extend(item.split(Path.SEPERATOR))
            else:
                self._items.append(item)
        return self

    def parent(self) -> _TPath:
        return Path(self._items[:-1])

    def __len__(self):
        return len(self._items)

    def __iter__(self) -> None:
        yield from self._items

    def __true_div__(self, other) -> _TPath:
        return self.duplicate().append(other)

    def __add__(self, other) -> _TPath:
        return self.duplicate().append(other)

    def __iadd__(self, other) -> _TPath:
        return self.append(other)

    def __getitem__(self, idx):
        return self._items[idx]

    def __setitem__(self, idx, item):
        self._items[idx] = item

    def reset(self, p=[]):
        self._items = p

    @property
    def empty(self) -> bool:
        return self._items is None or len(self._items) == 0

    def as_list(self) -> list:
        return self._items

    def normalize(self) -> _TPath:
        if self._items is None:
            self._items = []
        elif isinstance(self._items, str):
            self._items = [self._items]
        elif isinstance(self._items, tuple):
            self._items = list(self._items)
        elif not isinstance(self._items, collections.abc.MutableSequence):
            self._items = [self._items]

        self._items = sum([d.split(Path.SEPERATOR) if isinstance(d, str) else [d] for d in self._items], [])
        return self

    @property
    def is_closed(self) -> bool:
        return len(self._items) > 0 and self._items[-1] is None
