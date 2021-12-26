import collections.abc
from copy import deepcopy
from typing import Sequence, TypeVar
from spdm.common.tags import _undefined_
_TPath = TypeVar("_TPath", bound="Path")


class Path(object):
    SEPERATOR = '.'

    def __init__(self, *args, **kwargs):
        self._items = [Path.parser(a) for a in args if a is not None]

    def __repr__(self):
        return "@"+Path.SEPERATOR.join([str(d) for d in self._items])

    @staticmethod
    def parser(path) -> list:
        if path in (_undefined_,  None):
            return None
        elif isinstance(path, str):
            s_list = path.split(Path.SEPERATOR)
            if len(s_list) == 1:
                return s_list[0]
            else:
                res = Path()
                res._items = s_list
                return res
        elif isinstance(path, set):
            return {Path.parser(item) for item in path}
        elif isinstance(path, collections.abc.Sequence):
            return [Path.parser(item) for item in path]
        elif isinstance(path, collections.abc.Mapping):
            return {Path.parser(k): v for k, v in path}
        else:
            return path

    def duplicate(self) -> _TPath:
        res = Path()
        res._items = deepcopy(self._items)
        return res

    def append(self, *args) -> _TPath:
        if len(args) == 0:
            return self
        self._items.extend([Path.parser(a) for a in args if a is not None])
        return self

    def parent(self) -> _TPath:
        return Path(self._items[:-1])

    def __bool__(self) -> bool:
        return not self.empty

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
        if isinstance(idx, slice):
            return Path(self._items[idx])
        else:
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
