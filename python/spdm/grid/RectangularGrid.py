import typing
from .Grid import Grid
from .StructuredMesh import StructuredMesh
from ..geometry.Box import Box
import numpy as np
from ..utils.typing import ArrayType


@Grid.register(["rectangular", "rect"])
class RectangularGrid(StructuredMesh):
    """ Rectangular grid
        矩形网格
    """

    def __init__(self, dim1: ArrayType, dim2: ArrayType,  **kwargs) -> None:
        super().__init__(shape=[len(dim1), len(dim2)], ndims=2, **kwargs)
        self._dims = [dim1, dim2]

    @property
    def dim1(self) -> np.ndarray: return self._dims[0]
    @property
    def dim2(self) -> np.ndarray: return self._dims[1]

    @property
    def points(self) -> typing.Tuple[ArrayType, ...]:
        return tuple(np.meshgrid(*self._dims))


__SP_EXPORT__ = RectangularGrid
