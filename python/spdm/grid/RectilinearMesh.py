import typing
from functools import cached_property

import numpy as np
from scipy.interpolate import interpolate

from ..geometry.Curve import Curve
from ..geometry.Line import Line
from ..geometry.Point import Point
from ..utils.logger import logger
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

    def __init__(self, *args, coords: typing.List[np.ndarray] = [],  **kwargs) -> None:
        if coords is None or len(coords) == 0:
            coords = [(np.linspace(0, 1, d) if isinstance(d, int) else d) for d in args]
        elif len(args) > 0:
            logger.warning(f"Ignore position arguments {args}")
        super().__init__(shape=[len(d) for d in coords], **kwargs)
        self._coords = coords

    @cached_property
    def bbox(self) -> typing.List[float]:
        return [*[d[0] for d in self._coords], *[d[-1] for d in self._coords]]

    @cached_property
    def dx(self) -> typing.List[float]:
        return [(d[-1]-d[0])/len(d) for d in self._coords]

    def axis(self, idx, axis=0):
        p0 = [d[0] for d in self._coords]
        p1 = [d[-1] for d in self._coords]
        p0[axis] = self._coords[axis][idx]
        p1[axis] = self._coords[axis][idx]

        try:
            res = Line(p0, p1, is_closed=self.cycle[axis])
        except ValueError as error:
            res = Point(*p0)
        return res

    @cached_property
    def xy(self) -> typing.Sequence[np.ndarray]:
        if self.ndims == 1:
            return [self._coords[0]]
        elif self.ndims == 2:
            return np.meshgrid(*self._coords, indexing="ij")
        else:
            raise NotImplementedError()

    def point(self, *idx):
        return [d[idx[s]] for s, d in enumerate(self._coords)]

    def interpolator(self, value,  **kwargs):
        if value.shape != self.shape:
            raise ValueError(f"{value.shape} {self.shape}")

        if self.ndims == 1:
            interp = interpolate.InterpolatedUnivariateSpline(self._coords[0], value,  **kwargs)
        elif self.ndims == 2:
            interp = interpolate.RectBivariateSpline(self._coords[0], self._coords[1], value,  **kwargs)
        else:
            raise NotImplementedError(f"NDIMS {self.ndims}>2")

        return interp

    @ cached_property
    def dl(self):
        dX = (np.roll(self.points[0], 1, axis=1) - np.roll(self.points[0], -1, axis=1))/2.0
        dY = (np.roll(self.points[1], 1, axis=1) - np.roll(self.points[1], -1, axis=1))/2.0
        return dX, dY
