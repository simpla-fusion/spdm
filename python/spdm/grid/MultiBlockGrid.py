import typing

from numpy.typing import ArrayLike, NDArray

from ..data.Function import Function
from ..data.HTree import HTree
from .Grid import Grid


class MultiBlockGrid(HTree[Grid]):

    def __init__(self, block_list: typing.List[typing.Any],  **kwargs) -> None:
        super().__init__(block_list, **kwargs)

        self._cond_list = cond_list
        self._grid_list: typing.List[Grid] = grid_list

    def interpolator(self, *args, **kwargs) -> Function:
        return self._cond_list


class MultiBlockFunction(Function):

    def __init__(self, block_list: typing.List[typing.Any], **kwargs) -> None:
        super().__init__(**kwargs)

        self._grid_list: MultiBlockGrid = cond_list
        self._data_list: typing.List[Grid] = grid_list

    def __call__(self, *args, **kwargs) -> ArrayLike | NDArray:

        return super().__call__(*args, **kwargs)
