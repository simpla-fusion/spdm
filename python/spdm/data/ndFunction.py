import typing
import numpy as np
from .Node import Node

_T = typing.TypeVar("_T")


class ndFunction(Node, typing.Generic[_T]):
    pass

    def __call__(self, *args: typing.Any, **kwds:  typing.Any) -> _T:
        return super().__call__(*args, **kwds)

    def __array__(self) -> np.ndarray:
        return np.asarray(np.NaN)
