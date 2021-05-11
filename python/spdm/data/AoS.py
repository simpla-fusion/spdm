

from functools import cached_property
from typing import (Any, Generic, Iterator, Mapping, MutableMapping,
                    MutableSequence, Sequence, TypeVar, Union, get_args)


from .Node import Dict, List, Node, _TKey, _TObject


class AoS(Node.LazyAccessor):
    __slots__ = "_pos",

    def __init__(self, *args, pos=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pos = pos

    @property
    def path(self):
        return self._path[1:]+[self._path[0]]


class SoA(Node.LazyAccessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def path(self):
        return self._path[-1]+self._path[:-1]
