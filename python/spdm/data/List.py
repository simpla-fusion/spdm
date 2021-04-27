import collections

from matplotlib.pyplot import loglog
from spdm.util.logger import logger
from spdm.data.Node import Node
import typing

_KT = typing.TypeVar('_KT')
_VT = typing.TypeVar('_VT')


class List(typing.MutableSequence[_VT], Node):
    def __init__(self, d=None, *args, default_factory=None,  parent=None, **kwargs):
        Node.__init__(self, [], *args,  parent=parent, **kwargs)

        #  @ref: https://stackoverflow.com/questions/48572831/how-to-access-the-type-arguments-of-typing-generic?noredirect=1
        self._default_factory = default_factory or typing.get_args(self._orig_class)

    def __len__(self) -> int:
        return Node.__len__(self)

    def __setitem__(self, k: _KT, v: _VT) -> None:
        Node.__raw_set__(self, k, v)

    def __getitem__(self, k: _KT) -> _VT:
        return self._default_factory(Node.__raw_get__(self, k), parent=self._parent)
