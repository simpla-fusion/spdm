import typing

import numpy as np
from numpy.typing import NDArray, ArrayLike
from spdm.data.Function import Function
from spdm.data.Node import Node

from .Container import Container

_T = typing.TypeVar("_T")


class Profile(Function, Node,  typing.Generic[_T]):

    def __init__(self, *args, **kwargs) -> None:
        Node.__init__(self, *args, **{k: v for k, v in kwargs.items() if not k.startswith("coordinate")})
        if isinstance(self._parent, Container):
            coord_keys = [k for k in kwargs.keys() if k.startswith("coordinate")]
            coord_keys.sort()
            coord_keys = [kwargs[c] for c in coord_keys]

            # FIXME: "1...N" is for IMAS dd
            self._axis = [(slice(None) if (c == "1...N") else self._find_node_by_path(c)) for c in self._axis]

            Function.__init__(self, self,  *self._axis)
        else:
            raise RuntimeError(f"Parent is None, can not determint the coordinates!")

    @property
    def data(self) -> np.ndarray:
        return super().__value__()

    def __array__(self) -> NDArray | ArrayLike: return self.__call__(*self._axis)

    def __value__(self) -> ArrayLike | NDArray: return self.__array__()
    """aslias of __array__ """
