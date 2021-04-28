import collections
import typing

from ..util.logger import logger
from .Node import Node
from .Entry import Entry, _next_, _last_, _not_found_, _NEXT_TAG_

TKey = typing.TypeVar('TKey', int, slice,_NEXT_TAG_)
TValue = typing.TypeVar('TValue')


class List(typing.MutableSequence[TValue], Node):
    def __init__(self, d=None, *args, default_factory=None,  parent=None, **kwargs):
        Node.__init__(self, d or [], *args,  parent=parent, **kwargs)
        #  @ref: https://stackoverflow.com/questions/48572831/how-to-access-the-type-arguments-of-typing-generic?noredirect=1
        self._default_factory = default_factory or typing.get_args(getattr(self, "_orig_class", None)) or Node

    def __len__(self) -> int:
        return Node.__len__(self)

    def __setitem__(self, k: TKey, v: TValue) -> None:
        Node.__raw_set__(self, k, v)

    def __getitem__(self, k: TKey) -> TValue:
        return self._default_factory(Node.__raw_get__(self, k), parent=self._parent)

    def __delitem__(self, k: TKey) -> None:
        Node.__delitem__(self, k)

    def insert(self, *args, **kwargs):
        return Node.__raw_set__(self, *args, **kwargs)
