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


__SP_EXPORT__ = RectangularGrid
