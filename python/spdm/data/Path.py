from typing import TypeVar
import collections.abc
_TPath = TypeVar("_TPath", bound="Path")


class Path(object):
    SEPERATOR = '.'

    def __init__(self, d=None, *args, **kwargs):
        self._path = list(d) if d is not None else []

    def __repr__(self):
        return Path.SEPERATOR.join([str(d) for d in self._path])

    def append(self, *args) -> _TPath:
        for item in args:
            if isinstance(item, str):
                self._path.extend(item.split(Path.SEPERATOR))
            else:
                self._path.append(item)
        return self

    def parent(self) -> _TPath:
        return Path(self._path[:-1])

    def __len__(self):
        return len(self._path)

    def __iter__(self) -> None:
        yield from self._path

    def __true_div__(self, other) -> _TPath:
        return Path(self._path).append(other)

    def __add__(self, other) -> _TPath:
        return Path(self._path).append(other)

    def __getitem__(self, idx):
        return self._path[idx]

    def __setitem__(self, idx, item):
        self._path[idx] = item

    def empty(self) -> bool:
        return len(self._path) == 0

    def as_list(self) -> list:
        return self._path

    def normalize(self) -> _TPath:
        if self._path is None:
            self._path = []
        elif isinstance(self._path, str):
            self._path = [self._path]
        elif isinstance(self._path, tuple):
            self._path = list(self._path)
        elif not isinstance(self._path, collections.abc.MutableSequence):
            self._path = [self._path]

        self._path = sum([d.split(Path.SEPERATOR) if isinstance(d, str) else [d] for d in self._path], [])
        return self

    @property
    def is_closed(self) -> bool:
        return len(self._path) > 0 and self._path[-1] is None
