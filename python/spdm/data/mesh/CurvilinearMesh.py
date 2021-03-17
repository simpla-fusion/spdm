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

    def __init__(self,  *args, name="U,V", ** kwargs) -> None:
        if len(args) == 0:
            raise ValueError(f"Illegal input! ")
        elif not all([isinstance(a, np.ndarray) for a in args]):
            raise TypeError(f"{[type(a ) for a in args]} ")
        elif not all([a.shape == args[0].shape for a in args]):
            raise ValueError(f"Illegal shape! [{[a.shape for a in args]}]")

        super().__init__(shape=args[0].shape, name=name, **kwargs)
        self._points = args

    def axis(self, idx, axis=0):
        s = [slice(None, None, None)]*self.ndims
        s[axis] = idx
        s = tuple(s)
        try:
            res = BSplineCurve(*[p[s] for p in self.points],  is_closed=self.cycle[axis])
        except ValueError as error:
            logger.warning(f"Failed to create BSplineCurve: {error}")
            res = Point(*[p[0] for p in self.points])
        return res

    @cached_property
    def boundary(self):
        return PhysicalGraph({"inner": self.axis[0][0],  "outer": self.axis[0][-1]})

    @cached_property
    def bbox(self):
        return [[np.min(p) for p in self._points], [np.max(p) for p in self._points]]

    @cached_property
    def dl(self):
        return [NotImplemented]*self.ndims

    @property
    def points(self):
        return self._points

    @property
    def point(self, *idx):
        return [p[tuple(idx)] for p in self._points]
