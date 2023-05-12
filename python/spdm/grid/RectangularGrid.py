import typing
from .Grid import Grid
from .StructuredMesh import StructuredMesh
from ..geometry.Box import Box
import numpy as np
from ..utils.typing import ArrayType
from ..utils.logger import logger

@Grid.register(["rectangular", "rect"])
class RectangularGrid(StructuredMesh):
    """ Rectangular grid
        矩形网格
    """

    def __init__(self, dim1: ArrayType, dim2: ArrayType, *args, **kwargs) -> None:
        super().__init__(shape=[len(dim1) if dim1 is not None else 0, len(dim2) if dim2 is not None else 0], ndims=2, **kwargs)
        self._dims = [dim1, dim2]
        if len(args)>0:
           raise RuntimeError(f"Unexpected positional arguments: {args}")

    @property
    def dim1(self) -> np.ndarray: return self._dims[0]

    @property
    def dim2(self) -> np.ndarray: return self._dims[1]

    @property
    def points(self) -> typing.Tuple[ArrayType, ...]:
        return tuple(np.meshgrid(*self._dims))


__SP_EXPORT__ = RectangularGrid
