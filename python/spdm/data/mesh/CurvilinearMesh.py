from functools import cached_property, lru_cache
from math import log

import numpy as np
import scipy.interpolate

from ...util.logger import logger
from ..geometry.BSplineSurface import BSplineSurface
from ..geometry.CubicSplineCurve import CubicSplineCurve
from ..geometry.Point import Point
from ..PhysicalGraph import PhysicalGraph
from .StructuredMesh import StructuredMesh


class CurvilinearMesh(StructuredMesh):
    """
        A `curvilinear grid` or `structured grid` is a grid with the same combinatorial structure as a regular grid,
        in which the cells are quadrilaterals or [general] cuboids, rather than rectangles or rectangular cuboids.
            -- [https://en.wikipedia.org/wiki/Regular_grid]
    """
    TOLERANCE = 1.0e-5

    def __init__(self, xy, uv=None,  *args,   ** kwargs) -> None:

        super().__init__(*args, shape=xy[0].shape, rank=len(xy[0].shape), ndims=len(xy), **kwargs)

        if not all(x.shape == self.shape for x in xy):
            raise ValueError(f"illegal shape! uv={self.shape}  xy={[x.shape  for x in xy]} ")

        uv = uv if uv is not None else ([[0.0, 1.0]] * self.rank)

        for axis in range(self.rank):
            u = uv[axis]
            if isinstance(u, np.ndarray):
                assert(len(u) == self.shape[axis])
                continue

            if len(u) == 2:
                u = np.linspace(*u,  self.shape[axis])
            elif u is None:
                u = np.linspace(0.0, 1.0,  self.shape[axis])
            else:
                raise ValueError(f"Illegal {u}")

            uv[axis] = u

        self._xy = np.stack(xy)
        self._uv = uv

    def axis(self, idx, axis=0):
        s = [slice(None, None, None)]*self.ndims
        s[axis] = idx
        sub_xy = [p[tuple(s)] for p in self._xy]
        sub_uv = [self._uv[(axis+i) % self.ndims] for i in range(1, self.ndims)]
        sub_cycle = [self.cycle[(axis+i) % self.ndims] for i in range(1, self.ndims)]
        return CurvilinearMesh(sub_xy, sub_uv,  cycle=sub_cycle)

    @property
    def xy(self):
        return self._xy

    @property
    def uv(self):
        return self._uv

    # def pushforward(self, new_uv):
    #     new_shape = [len(u) for u in new_uv]
    #     if new_shape != self.shape:
    #         raise ValueError(f"illegal shape! {new_shape}!={self.shape}")
    #     return CurvilinearMesh(self._xy, new_uv, cycle=self.cycle)

    def interpolator(self, value,  **kwargs):
        if value.shape != self.shape:
            raise ValueError(f"{value.shape} {self.shape}")

        if self.ndims == 1:
            interp = scipy.interpolate.InterpolatedUnivariateSpline(self._dims[0], value,  **kwargs)
        elif self.ndims == 2:
            interp = scipy.interpolate.RectBivariateSpline(self._dims[0], self._dims[1], value, ** kwargs)
        else:
            raise NotImplementedError(f"NDIMS {self.ndims}>2")
        return interp

    @cached_property
    def boundary(self):
        return PhysicalGraph({"inner": self.axis(0, 0),  "outer": self.axis(-1, 0)})

    @cached_property
    def geo_object(self):
        if self.rank == 1:
            if all([np.var(x)/np.mean(x**2) < CurvilinearMesh.TOLERANCE for x in self._xy]):
                gobj = Point(*[x[0] for x in self._xy])
            else:
                gobj = CubicSplineCurve(self._uv[0], self._xy, is_closed=self.cycle[0])
        elif self.rank == 2:
            gobj = BSplineSurface(self._uv, self._xy, is_closed=self.cycle)
        else:
            raise NotImplementedError()

        return gobj
