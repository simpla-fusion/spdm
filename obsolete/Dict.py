import collections
import typing

from spdm.numlib import np

from .List import List
from .Node import Node, _TObject, _TKey


class Dict(Node[_TKey, _TObject], typing.MutableMapping[_TKey, _TObject]):
    def __init__(self, data: typing.Mapping = {}, *args,  **kwargs):
        Node.__init__(self, data, *args,   **kwargs)

    def __post_process__(self, value, *args, **kwargs):
        if isinstance(value, Node):
            return value
        elif isinstance(value, (str, int, float, np.ndarray)):
            return value
        elif isinstance(value, collections.abc.MutableSequence):
            return List[_TObject](value, parent=self)
        elif isinstance(value, collections.abc.MutableMapping):
            return Dict[_TKey, _TObject](value, parent=self)
        else:
            return Node(value, parent=self)

    def __pre_process__(self, value, *args, **kwargs):
        return Node.__pre_process__(self, value, *args, **kwargs)

    def __getitem__(self, key: _TKey) -> _TObject:
        return self.__post_process__(self.__raw_get__(key))

    def __setitem__(self, key: _TKey, value: _TObject) -> None:
        self.__raw_set__(key, self.__pre_process__(value))

    def __delitem__(self, v: _TKey) -> None:
        return Node.__delitem__(self, v)

    def __iter__(self) -> typing.Iterator[Node]:
        return Node.__iter__(self)

    def __len__(self) -> int:
        return Node.__len__(self)

    def __eq__(self, o: object) -> bool:
        return Node.__eq__(self, o)

    def __contains__(self, o: object) -> bool:
        return Node.__contains__(self, o)
