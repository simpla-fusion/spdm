import typing
from functools import cached_property

import numpy as np
from scipy.interpolate import interpolate

from ..data.Function import Function
from ..geometry.Box import Box
from ..geometry.Curve import Curve
from ..geometry.GeoObject import GeoObject
from ..geometry.Line import Line
from ..geometry.Point import Point
from ..utils.logger import logger
from ..utils.typing import ArrayType, NumericType, ScalarType
from .Mesh import Mesh
from .StructuredMesh import StructuredMesh


@Mesh.register("rectilinear")
class RectilinearMesh(StructuredMesh):
    """
        A `rectilinear Mesh` is a tessellation by rectangles or rectangular cuboids (also known as rectangular parallelepipeds)
        that are not, in general, all congruent to each other. The cells may still be indexed by integers as above, but the
        mapping from indexes to vertex coordinates is less uniform than in a regular Mesh. An example of a rectilinear Mesh
        that is not regular appears on logarithmic scale graph paper.
            -- [https://en.wikipedia.org/wiki/Regular_Mesh]

        RectlinearMesh
        --------------------
        可以视为由 n=rank 条称为axis的曲线 curve 平移张成的空间。

        xyz= sum([ axis[i](uvw[i]) for i in range(rank) ])

    """

    def __init__(self, *dims: ArrayType, geometry=None, **kwargs) -> None:

        if geometry is None:
            geometry = Box([min(d) for d in dims], [max(d) for d in dims])
        super().__init__(shape=[len(d) for d in dims], geometry=geometry, **kwargs)
        self._dims = dims
        self._aixs: typing.List[Curve] = [
            Function(self._dims[i], np.linspace(0, 1.0, self.shape[i])) for i in range(self.rank)]

    @property
    def dim1(self) -> ArrayType: return self._dims[0].__array__()

    @property
    def dim2(self) -> ArrayType: return self._dims[1].__array__()

    @property
    def rank(self) -> int: return len(self._dims)

    @cached_property
    def bbox(self) -> typing.Tuple[ArrayType, ArrayType]:
        return tuple([[d[0] for d in self._dims], [d[-1] for d in self._dims]])

    @cached_property
    def dx(self) -> ArrayType: return np.asarray([(d[-1]-d[0])/len(d) for d in self._dims])

    def coordinates(self, *uvw) -> ArrayType:
        """ 网格点的 _空间坐标_
            @return: _数组_ 形状为 [geometry.dimension,<shape of uvw ...>]
        """
        if len(uvw) == 1 and self.rank != 1:
            uvw = uvw[0]
        return np.stack([self._dims[i](uvw[i]) for i in range(self.rank)])

    @cached_property
    def vertices(self) -> ArrayType:
        """ 网格点的 _空间坐标_ """
        if self.geometry.rank == 1:
            return (self._dims[0],)
        elif self.geometry.rank == 2:
            return np.stack(np.meshgrid(*self._dims, indexing="ij"))
        else:
            raise NotImplementedError()

    @property
    def points(self) -> ArrayType: return self.vertices
    """ alias of vertices """

    def interpolator(self, value,  **kwargs):
        if np.any(tuple(value.shape) != tuple(self.shape)):
            raise ValueError(f"{value.shape} {self.shape}")

        if self.geometry.ndim == 1:
            interp = interpolate.InterpolatedUnivariateSpline(self._dims[0], value,  **kwargs)
        elif self.geometry.ndim == 2:
            interp = interpolate.RectBivariateSpline(self._dims[0], self._dims[1], value,  **kwargs)
        else:
            raise NotImplementedError(f"NDIMS={self.geometry.ndim}")

        return interp
