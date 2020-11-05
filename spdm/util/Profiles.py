
import collections
from copy import copy

import numpy as np

from .LazyProxy import LazyProxy
from .logger import logger


class Profiles(dict):
    """ Collection of '''profile'''s with common '''dimension'''.

        dims likes [0,1,2,3,4,5]
            or [
                 [0,1,2,3,4,5]
            ]
    """

    def __init__(self,  dims, *args, interpolator=None, ** kwargs):

        if isinstance(dims, np.ndarray):
            if len(dims.shape) == 1:
                self._grid = dims
                self._grid_shape = dims.shape[0]
                self._axis_rank = 1
            else:
                self._grid = [x for x in dims]
                self._grid_shape = dims.shape[1:]
                self._axis_rank = dims.shape[0]

        # elif not isinstance(dims, list):
        #     raise TypeError(f"Illegal dimensions type {type(dims)}! ( only support list or numpy.ndarray)")
        # elif all([type(p) is int for p in dims]):
        #     self._grid = np.array(dims)
        #     self._grid_shape = len(dims)
        #     self._axis_rank = 1
        # elif all([isinstance(p, np.ndarray) for p in dims]):
        #     self._grid = dims
        #     self._grid_shape = dims[0].shape
        #     self._axis_rank = len(dims)

        self._interpolator = interpolator

    @property
    def entry(self):
        return LazyProxy(self)

    @property
    def grid(self):
        return self._grid

    @property
    def grid_shape(self):
        return self._grid_shape

    @property
    def axis_rank(self):
        return self._axis_rank

    def copy(self):
        return copy(self)

    def count(self, path):
        return super().__contains__(".".join(path))

    def get(self, path):
        return super().__getitem__(".".join(path))

    def get_value(self, path):
        return super().__getitem__(".".join(path))

    def put(self, path, value):
        return super().__setitem__(".".join(path), value)

    def interploate(self, path, *args, interpolator=None, **kwargs):
        obj = self.get(path)
        if len(args) is 0:
            return obj

        interpolator = interpolator or self._interpolator

        if interpolator is None:
            raise RuntimeError("Interploator is not defined")

        return interpolator(self.grid, obj, *args, **kwargs)

    def call(self,  path, *args, **kwargs):
        return self.interploate(path, *args, **kwargs)

    def __missing__(self, key):
        return super().setdefault(key, np.zeros(shape=self.grid_shape))


class Profiles1D(Profiles):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)


class Profiles2D(Profiles):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
