import typing
from .Mesh import Mesh
from .mesh_structured import StructuredMesh
import numpy as np
from ..utils.typing import ArrayType
from ..utils.logger import logger
from .mesh_rectilinear import RectilinearMesh


@Mesh.register(["rectangular", "rect"])
class RectangularMesh(RectilinearMesh):
    """ Rectangular Mesh, which is alias of RectilinearMesh
        矩形网格
    """

    @property
    def dim1(self) -> ArrayType: return self._dims[0]

    @property
    def dim2(self) -> ArrayType: return self._dims[1]


