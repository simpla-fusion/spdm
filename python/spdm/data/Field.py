from __future__ import annotations

import os
import typing
from enum import Enum
from functools import lru_cache

import numpy as np

from ..grid.Grid import Grid
from ..utils.logger import logger
from ..utils.misc import regroup_dict_by_prefix
from ..utils.typing import ArrayType, NumericType
from .Function import Function
from .Node import Node
from .Profile import Profile

_T = typing.TypeVar("_T")


class Field(Node, Function, typing.Generic[_T]):
    """ Field
        ---------
        A field is a function that assigns a value to every point of space. The value of the field can be a scalar or a vector.

        一个 _场(Field)_ 是一个 _函数(Function)_，它为空间(默认为多维流形)上的每一点分配一个值。场的值可以是标量或矢量。
    """

    def __init__(self, *args, **kwargs):
        grid, kwargs = regroup_dict_by_prefix(kwargs, "grid")
        coordinates, kwargs = regroup_dict_by_prefix(kwargs, "coordinate", keep_prefix=True, sep='')
        super().__init__(*args, **kwargs)

        if isinstance(grid, Grid):
            pass
        elif isinstance(grid, dict) and len(grid) > 0:
            grid = Grid(**grid)
        elif len(coordinates) > 0:
            # 获得 coordinates 中 value的共同前缀
            prefix = os.path.commonprefix([*coordinates.values()])
            if prefix.startswith("../grid/"):
                grid = getattr(self._parent, "grid", None)
                if isinstance(grid, Node):
                    grid_type = getattr(self._parent, "grid_type", None)
                    logger.debug(grid.dim1)
                    grid = Grid(grid.dim1, grid.dim2, grid.volume_element, type=grid_type)
            else:
                logger.debug(coordinates)
                grid = Grid()

        if not isinstance(grid, Grid):
            grid_type = getattr(self._parent, "grid_type", None)

            if not isinstance(grid_type, str):
                grid_type = getattr(grid_type, "name", None)

            grid = Grid(grid, type=grid_type)

        Function.__init__(self, grid=grid)

    @ property
    def metadata(self) -> typing.Mapping[str, typing.Any]:
        return super(Node, self).metadata

    @ property
    def data(self) -> np.ndarray:
        return super().__value__()

    def __array__(self) -> NDArray | ArrayLike: return self.__call__(*self._axis)

    def __value__(self) -> ArrayLike | NDArray: return self.__array__()
    """aslias of __array__ """

    def plot(self, axis, *args,  **kwargs):

        kwargs.setdefault("linewidths", 0.1)

        axis.contour(*self._grid.points,  self.__array__(), **kwargs)

        return axis
