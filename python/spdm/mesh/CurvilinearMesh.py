from functools import cached_property, lru_cache
from math import log

import numpy as np
import scipy.interpolate

from ..geometry.BSplineSurface import BSplineSurface
from ..geometry.CubicSplineCurve import CubicSplineCurve
from ..geometry.Point import Point
from ..util.logger import logger
from .StructuredMesh import StructuredMesh
from ..util.utilities import convert_to_named_tuple


class CurvilinearMesh(StructuredMesh):
    """
        A `curvilinear grid` or `structured grid` is a grid with the same combinatorial structure as a regular grid,
        in which the cells are quadrilaterals or [general] cuboids, rather than rectangles or rectangular cuboids.
            -- [https://en.wikipedia.org/wiki/Regular_grid]
    """
    TOLERANCE = 1.0e-5

    def __init__(self, xy, uv=None,  *args,   ** kwargs) -> None:
        super().__init__(*args, shape=[len(d) for d in uv], rank=len(uv), ndims=xy.shape[-1], **kwargs)
        self._xy = xy.reshape([*self.shape, self.ndims])
        self._uv = uv

    def axis(self, idx, axis=0):
        s = [slice(None, None, None)]*self.ndims
        s[axis] = idx
        s = s+[slice(None, None, None)]

        sub_xy = self._xy[tuple(s)]  # [p[tuple(s)] for p in self._xy]
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
        return convert_to_named_tuple({"inner": self.axis(0, 0),  "outer": self.axis(-1, 0)})

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
