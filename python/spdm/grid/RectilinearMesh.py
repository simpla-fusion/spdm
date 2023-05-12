import typing
from functools import cached_property

import numpy as np
from scipy.interpolate import interpolate

from ..geometry.Curve import Curve
from ..geometry.Line import Line
from ..geometry.Box import Box
from ..geometry.Point import Point

from ..utils.logger import logger
from ..utils.typing import ArrayType, ScalarType
from .Grid import Grid
from .StructuredMesh import StructuredMesh


@Grid.register("rectlinear")
class RectilinearMesh(StructuredMesh):
    """
        A `rectilinear grid` is a tessellation by rectangles or rectangular cuboids (also known as rectangular parallelepipeds)
        that are not, in general, all congruent to each other. The cells may still be indexed by integers as above, but the
        mapping from indexes to vertex coordinates is less uniform than in a regular grid. An example of a rectilinear grid
        that is not regular appears on logarithmic scale graph paper.
            -- [https://en.wikipedia.org/wiki/Regular_grid]

    """

    def __init__(self, *dims: ArrayType,  **kwargs) -> None:
        if len(dims) == 0:
            dims = kwargs.get("dims", [])
        if "geometry" not in kwargs:
            kwargs["geometry"] = Box([min(d) for d in dims], [max(d) for d in dims])
        super().__init__(shape=[len(d) for d in dims], **kwargs)

        self._dims = dims

    @property
    def dim1(self) -> ArrayType: return self._dims[0]

    @property
    def dim2(self) -> ArrayType: return self._dims[1]

    @cached_property
    def bbox(self) -> typing.List[float]:
        return [*[d[0] for d in self._dims], *[d[-1] for d in self._dims]]

    @cached_property
    def dx(self) -> typing.List[float]:
        return [(d[-1]-d[0])/len(d) for d in self._dims]

    def axis(self, idx, axis=0):
        p0 = [d[0] for d in self._dims]
        p1 = [d[-1] for d in self._dims]
        p0[axis] = self._dims[axis][idx]
        p1[axis] = self._dims[axis][idx]

        try:
            res = Line(p0, p1, is_closed=self.cycle[axis])
        except ValueError as error:
            res = Point(*p0)
        return res

    @cached_property
    def points(self) -> typing.Tuple[ArrayType, ...]:
        if self.geometry.ndims == 1:
            return (self._dims[0],)
        elif self.geometry.ndims == 2:
            return tuple(np.meshgrid(*self._dims, indexing="ij"))
        else:
            raise NotImplementedError()

    def interpolator(self, value,  **kwargs):
        if np.any(value.shape != self.shape):
            raise ValueError(f"{value.shape} {self.shape}")

        if self.geometry.ndims == 1:
            interp = interpolate.InterpolatedUnivariateSpline(self._dims[0], value,  **kwargs)
        elif self.geometry.ndims == 2:
            interp = interpolate.RectBivariateSpline(self._dims[0], self._dims[1], value,  **kwargs)
        else:
            raise NotImplementedError(f"NDIMS {self.geometry.ndims}>2")

        return interp

    @cached_property
    def dl(self):
        dX = (np.roll(self.points[0], 1, axis=1) - np.roll(self.points[0], -1, axis=1))/2.0
        dY = (np.roll(self.points[1], 1, axis=1) - np.roll(self.points[1], -1, axis=1))/2.0
        return dX, dY
