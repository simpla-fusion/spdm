from functools import cached_property

import numpy as np
import scipy.interpolate
from numpy.lib.function_base import interp, meshgrid

from ..geometry.BSplineCurve import BSplineCurve
from ..Mesh import Mesh
from ..PhysicalGraph import PhysicalGraph


class StructedMesh2D(Mesh):
    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args,   **kwargs)

    def axis(self, idx, axis=0):
        return NotImplemented

    @cached_property
    def dl(self):
        return NotImplemented, NotImplemented

    @cached_property
    def xy(self):
        return NotImplemented, NotImplemented


class RectilinearMesh(StructedMesh2D):
    """
        A `rectilinear grid` is a tessellation by rectangles or rectangular cuboids (also known as rectangular parallelepipeds)
        that are not, in general, all congruent to each other. The cells may still be indexed by integers as above, but the 
        mapping from indexes to vertex coordinates is less uniform than in a regular grid. An example of a rectilinear grid 
        that is not regular appears on logarithmic scale graph paper.
            -- [https://en.wikipedia.org/wiki/Regular_grid]

    """

    def __init__(self, *args, **kwargs) -> None:
        def normalize_dim(d):
            if isinstance(d, np.ndarray):
                return d
            elif isinstance(d, int):
                return np.linspace(0, 1, d)
            else:
                raise TypeError(type(d))

        self._axis = [normalize_dim(d) for d in args]
        super().__init__(*args, shape=tuple([len(d) for d in self._axis]),  **kwargs)

    @property
    def axis(self):
        return self._axis

    @property
    def bbox(self):
        return [[d[0], d[-1]] for d in self._axis]

    def point(self, *idx):
        return [m[tuple(idx)] for m in self.mesh]

    def interpolator(self, value,  **kwargs):
        assert(value.shape == self.shape)
        if self.ndims == 1:
            interp = scipy.interpolate.InterpolatedUnivariateSpline(self._axis[0], value,  **kwargs)
        elif self.ndims == 2:
            interp = scipy.interpolate.RectBivariateSpline(self._axis[0], self._axis[1], value, ** kwargs)
        else:
            raise NotImplementedError(f"NDIMS {self.ndims}>2")

        return interp

    @cached_property
    def dl(self):
        dX = (np.roll(self.xy[0], 1, axis=1) - np.roll(self.xy[0], -1, axis=1))/2.0
        dY = (np.roll(self.xy[1], 1, axis=1) - np.roll(self.xy[1], -1, axis=1))/2.0
        return dX, dY

    @cached_property
    def xy(self):
        return np.meshgrid(*self._axis, indexing="ij")


class CurvilinearMesh2D(StructedMesh2D):
    """
        A `curvilinear grid` or `structured grid` is a grid with the same combinatorial structure as a regular grid,
        in which the cells are quadrilaterals or [general] cuboids, rather than rectangles or rectangular cuboids.
            -- [https://en.wikipedia.org/wiki/Regular_grid]
    """

    def __init__(self, X, Y, *args, name="U,V", ** kwargs) -> None:
        if not (isinstance(X, np.ndarray) and isinstance(Y, np.ndarray)):
            raise TypeError(f"{type(X)} ,  {type(Y)}")
        elif not (X.shape == Y.shape) or len(X.shape) != 2:
            raise ValueError(f"Illegal shape! [{X.shape} , {Y.shape}]")
        super().__init__(*args, shape=X.shape, name=name, **kwargs)
        self._xy = X, Y

    def axis(self, idx, axis=0):
        if axis == 0:
            res = BSplineCurve(self.xy[0][idx, :], self.xy[1][idx, :], cycle=self.cycle[0])
        else:
            res = BSplineCurve(self.xy[0][:, idx], self.xy[1][:, idx], cycle=self.cycle[1])

        return res

    @cached_property
    def boundary(self):
        return PhysicalGraph({"inner": self.axis[0][0],  "outer": self.axis[0][-1]})

    @cached_property
    def bbox(self):
        return [[np.min(self._xy[0]), np.min(self._xy[1])], [np.max(self._xy[0]), np.max(self._xy[1])]]

    @cached_property
    def dl(self):
        return NotImplemented, NotImplemented

    @cached_property
    def xy(self):
        return self._xy
