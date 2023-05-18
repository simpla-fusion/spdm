import pprint
import typing
from functools import cached_property
import numpy as np
from ..utils.misc import group_dict_by_prefix
from ..utils.typing import ArrayType, NumericType
from ..utils.tags import _not_found_
from .Container import Container
from .Expression import Expression
from .Function import Function
from .Node import Node

_T = typing.TypeVar("_T")


class Profile(Node, Function[_T]):

    def __init__(self,  value: NumericType | Expression, *dims, **kwargs) -> None:

        if isinstance(value, Expression):
            expr = value
            value = None
        else:
            expr = None

        Node.__init__(self, value, **kwargs)

        coordinats = {int(k[10:]): v for k, v in self._metadata.items() if k.startswith("coordinate")}

        if len(coordinats) > 0:
            if not isinstance(self._parent, Container):
                raise RuntimeError(f"Parent is None, can not determint the coordinates!")

            coord_path = [*coordinats.keys()]
            coord_path.sort()
            coordinats = [coordinats[c] for c in coord_path]

            # FIXME: "1...N" is for IMAS dd
            domain = tuple([(slice(None) if (c == "1...N") else self._find_node_by_path(c)) for c in coordinats])
        else:
            domain = dims

        Function.__init__(self, expr, *domain)

    def __str__(self) -> str: return Function.__str__(self)

    @property
    def name(self) -> str: return self._metadata.get("name", 'unnamed')

    @property
    def data(self) -> ArrayType: return self.__array__()

    def __value__(self) -> ArrayType:
        if not isinstance(self._value, np.ndarray) and not self._value:
            self._value = Node.__value__(self)
        return self._value
