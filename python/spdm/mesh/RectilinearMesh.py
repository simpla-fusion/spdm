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


@Mesh.register("rectlinear")
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
    def dim1(self) -> ArrayType: return self._dims[0]

    @property
    def dim2(self) -> ArrayType: return self._dims[1]

    @cached_property
    def bbox(self) -> ArrayType:
        return [*[d[0] for d in self._dims], *[d[-1] for d in self._dims]]

    @cached_property
    def dx(self) -> ArrayType: return np.asarray([(d[-1]-d[0])/len(d) for d in self._dims])

    def points(self, *uv) -> typing.Tuple[NumericType, ...]:
        shift = sum([(self._axis[i](uv[i])-self._axis[i].p0) for i in range(self.rank-1)])
        return self._axis[-1].translate(shift)(uv[-1])

    def plane(self, n_axis: int, uv: float) -> GeoObject:
        shit = self._axis[n_axis](uv)-self._axis[n_axis].p0
        return self.__class__(*[self._axis[(n_axis+i) % self.rank].translate(shit) for i in range(self.rank-1)])

    def axis(self, idx, axis=0) -> GeoObject:
        p0 = [d[0] for d in self._dims]
        p1 = [d[-1] for d in self._dims]
        p0[axis] = self._dims[axis][idx]
        p1[axis] = self._dims[axis][idx]

        try:
            res = Line(p0, p1, is_closed=self.cycle[axis])
        except ValueError as error:
            res = Point(*p0)
        return res

    @ cached_property
    def points(self) -> typing.Tuple[ArrayType, ...]:
        if self.geometry.ndim == 1:
            return (self._dims[0],)
        elif self.geometry.ndim == 2:
            return tuple(np.meshMesh(*self._dims, indexing="ij"))
        else:
            raise NotImplementedError()

    def interpolator(self, value,  **kwargs):
        if np.any(value.shape != self.shape):
            raise ValueError(f"{value.shape} {self.shape}")

        if self.geometry.ndim == 1:
            interp = interpolate.InterpolatedUnivariateSpline(self._dims[0], value,  **kwargs)
        elif self.geometry.ndim == 2:
            interp = interpolate.RectBivariateSpline(self._dims[0], self._dims[1], value,  **kwargs)
        else:
            raise NotImplementedError(f"NDIMS {self.geometry.ndim}>2")

        return interp

    @ cached_property
    def dl(self):
        dX = (np.roll(self.points[0], 1, axis=1) - np.roll(self.points[0], -1, axis=1))/2.0
        dY = (np.roll(self.points[1], 1, axis=1) - np.roll(self.points[1], -1, axis=1))/2.0
        return dX, dY
