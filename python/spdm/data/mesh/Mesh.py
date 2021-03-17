import collections
from functools import cached_property

import numpy as np
import scipy.interpolate

from ...util.logger import logger
from ...util.SpObject import SpObject
from ..Unit import Unit


class Mesh(SpObject):

    @staticmethod
    def __new__(cls, *args, mesh_type=None, grid_index=0, **kwargs):
        if cls is not Mesh:
            return object.__new__(cls)

        n_cls = None
        if mesh_type is None or mesh_type == "rectilinear" or grid_index == 0:
            n_cls = RectilinearMesh
        else:
            raise NotImplementedError()

        return object.__new__(n_cls)

    def __init__(self, *args, ndims=None, shape=None, name=None, unit=None, cycle=None, **kwargs) -> None:

        self._shape = shape or []
        self._ndims = ndims or len(shape or [])

        name = name or [""] * self._ndims
        if isinstance(name, str):
            self._name = name.split(",")
        elif not isinstance(name, collections.abc.Sequence):
            self._name = [name]

        unit = unit or [None] * self._ndims
        if isinstance(unit, str):
            unit = unit.split(",")
        elif not isinstance(unit, collections.abc.Sequence):
            unit = [unit]
        if len(unit) == 1:
            unit = unit * self._ndims
        # self._unit = [*map(Unit(u for u in unit))]

        cycle = cycle or [False] * self._ndims
        if not isinstance(cycle, collections.abc.Sequence):
            cycle = [cycle]
        if len(cycle) == 1:
            cycle = cycle * self._ndims
        self._cycle = cycle

    @property
    def name(self):
        return self._name

    @property
    def unit(self):
        return self._unit

    @property
    def cycle(self):
        return self._cycle

    @property
    def ndims(self):
        return len(self.shape)

    @property
    def shape(self):
        return self._shape

    @property
    def topology_rank(self):
        return self.ndims

    @cached_property
    def bbox(self):
        return NotImplemented

    @cached_property
    def boundary(self):
        return NotImplemented

    def new_dataset(self, *args, **kwargs):
        return np.ndarray(self._shape, *args, **kwargs)


class RectilinearMesh(Mesh):
    """
        A `rectilinear grid` is a tessellation by rectangles or rectangular cuboids (also known as rectangular parallelepipeds)
        that are not, in general, all congruent to each other. The cells may still be indexed by integers as above, but the 
        mapping from indexes to vertex coordinates is less uniform than in a regular grid. An example of a rectilinear grid 
        that is not regular appears on logarithmic scale graph paper.
            -- [https://en.wikipedia.org/wiki/Regular_grid]

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,  **kwargs)

        def normalize_dim(d):
            if isinstance(d, np.ndarray):
                return d
            elif isinstance(d, int):
                return np.linspace(0, 1, d)
            else:
                raise TypeError(type(d))

        self._axis = [normalize_dim(d) for d in args]

    @property
    def axis(self):
        return self._axis

    @property
    def ndims(self):
        return len(self._axis)

    @property
    def shape(self):
        return tuple([len(d) for d in self._axis])

    @property
    def bbox(self):
        return [[d[0], d[-1]] for d in self._axis]

    @cached_property
    def mesh(self):
        return np.meshgrid(*self._axis, indexing="ij")

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
