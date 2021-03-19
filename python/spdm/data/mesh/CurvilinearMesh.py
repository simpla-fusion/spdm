from functools import cached_property, lru_cache

import numpy as np

from ...util.logger import logger
from ..geometry.BSplineCurve import BSplineCurve
from ..geometry.Point import Point
from ..PhysicalGraph import PhysicalGraph
from .StructedMesh import StructedMesh


class CurvilinearMesh(StructedMesh):
    """
        A `curvilinear grid` or `structured grid` is a grid with the same combinatorial structure as a regular grid,
        in which the cells are quadrilaterals or [general] cuboids, rather than rectangles or rectangular cuboids.
            -- [https://en.wikipedia.org/wiki/Regular_grid]
    """
    TOLERANCE = 1.0e-5

    def __init__(self,   xy, uv=None,  *args, ** kwargs) -> None:
        if uv is None:
            uv = [np.linspace(0, 1.0, s) for s in xy[0].shape]
        shape = tuple([len(u) for u in uv])
        if not all(x.shape == shape for x in xy):
            raise ValueError(f"illegal shape! uv={shape}  xy={[x.shape  for x in xy]} ")

        super().__init__(*args, shape=shape, rank=len(uv), ndims=len(xy), **kwargs)
        self._xy = xy
        self._uv = uv

    def axis(self, idx, axis=0):
        s = [slice(None, None, None)]*self.ndims
        s[axis] = idx
        sub_xy = [p[tuple(s)] for p in self._xy]
        sub_uv = [self._uv[(axis+i) % self.ndims] for i in range(1, self.ndims)]
        sub_cycle = [self.cycle[(axis+i) % self.ndims] for i in range(1, self.ndims)]
        return CurvilinearMesh(sub_xy, sub_uv,  cycle=sub_cycle)

    def pushforward(self, new_uv):
        new_shape = [len(u) for u in new_uv]
        if new_shape != self.shape:
            raise ValueError(f"illegal shape! {new_shape}!={self.shape}")
        return CurvilinearMesh(self._xy, new_uv, cycle=self.cycle)

    @cached_property
    def boundary(self):
        return PhysicalGraph({"inner": self.axis(0, 0),  "outer": self.axis(-1, 0)})

    @cached_property
    def bbox(self):
        return [[np.min(p) for p in self._xy], [np.max(p) for p in self._xy]]

    @property
    def points(self):
        return self._xy

    @property
    def point(self, *idx):
        return [p[tuple(idx)] for p in self._xy]

    @cached_property
    def geo_object(self):
        if self.rank == 1:
            if all([np.var(x)/np.mean(x**2) < CurvilinearMesh.TOLERANCE for x in self._xy]):
                gobj = Point(*[x[0] for x in self._xy])
            else:
                gobj = BSplineCurve(self._uv[0], self._xy, cycle=self.cycle[0])
        elif self.rank == 2:
            gobj = BSplineCurve(self._uv, self._xy, cycle=self.cycle)

        return gobj
