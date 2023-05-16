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

    def __init__(self,  value: NumericType | Expression, **kwargs) -> None:

        domain, kwargs = group_dict_by_prefix(kwargs, "coordinate")

        if isinstance(value, Expression):
            expr = value
            value = None
        else:
            expr = None

        Node.__init__(self, value, **kwargs)

        if len(domain) > 0:
            if not isinstance(self._parent, Container):
                raise RuntimeError(f"Parent is None, can not determint the coordinates!")

            domain_keys = [*domain.keys()]
            domain_keys.sort()
            domain_keys = [domain[c] for c in domain_keys]

            # FIXME: "1...N" is for IMAS dd
            domain = tuple([(slice(None) if (c == "1...N") else self._find_node_by_path(c)) for c in domain_keys])

        Function.__init__(self, expr, *domain)

    @property
    def data(self) -> ArrayType: return self.__array__()

    def __value__(self) -> ArrayType: return self.__array__()

    def __array__(self) -> ArrayType:
        if not isinstance(self._value, np.ndarray) and not self._value:
            self._value = Node.__value__(self)
        return super().__array__()
