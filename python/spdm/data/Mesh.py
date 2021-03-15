from matplotlib.pyplot import loglog
from numpy.lib.function_base import interp, meshgrid
from ..util.SpObject import SpObject
from ..util.logger import logger
import numpy as np
from functools import cached_property
import scipy.interpolate
# import (RectBivariateSpline, SmoothBivariateSpline, UnivariateSpline)


class Mesh(SpObject):

    @staticmethod
    def __new__(cls, *args, grid_type=None, grid_index=0, **kwargs):
        if cls != Mesh:
            return super(Mesh, SpObject).__new__(None, *args, type=mesh_type, **kwargs)

        n_cls = None
        if grid_type is None or grid_type == "rectangle" or grid_index == 0:
            n_cls = RectangleMesh
        else:
            raise NotImplementedError()

        return object.__new__(n_cls)

    def __init__(self, *args, **kwargs) -> None:
        self._shape = []

    @property
    def axis(self):
        return NotImplemented

    @property
    def ndims(self):
        return NotImplemented

    @property
    def topology_rank(self):
        return self.ndims

    def new_dataset(self, *args, **kwargs):
        return np.ndarray(self._shape, *args, **kwargs)


class RectangleMesh(Mesh):
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
