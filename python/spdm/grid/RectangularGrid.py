import typing
from .Grid import Grid
from .StructuredMesh import StructuredMesh
from ..geometry.Box import Box
import numpy as np
from ..utils.typing import ArrayType
from ..utils.logger import logger
from .RectilinearMesh import RectilinearMesh


@Grid.register(["rectangular", "rect"])
class RectangularGrid(RectilinearMesh):
    """ Rectangular grid, which is alias of RectilinearMesh
        矩形网格
    """

    @property
    def dim1(self) -> ArrayType: return self._dims[0]

    @property
    def dim2(self) -> ArrayType: return self._dims[1]


__SP_EXPORT__ = RectangularGrid
