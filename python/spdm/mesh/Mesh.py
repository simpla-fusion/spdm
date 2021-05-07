import collections
from functools import cached_property

import numpy as np
import scipy.interpolate

from ..util.logger import logger
from ..util.SpObject import SpObject


class Mesh(SpObject):

    @staticmethod
    def __new__(cls, *args, mesh_type=None,   **kwargs):
        if cls is not Mesh:
            return object.__new__(cls)

        n_cls = None
        if mesh_type is None or mesh_type == "rectilinear":
            from .RectilinearMesh import RectilinearMesh
            n_cls = RectilinearMesh
        else:
            raise NotImplementedError(mesh_type)

        return object.__new__(n_cls)

    def __init__(self, *args, ndims=None, rank=None, shape=None, name=None, unit=None, cycle=None, **kwargs) -> None:
        self._rank = rank or len(shape or [])
        self._shape = shape or []
        self._ndims = ndims or self._rank

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

        # logger.debug(f"Create {self.__class__.__name__} rank={self.rank} shape={self.shape} ndims={self.ndims}")

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
        return self._ndims

    @property
    def rank(self):
        return self._rank

    @property
    def shape(self):
        return tuple(self._shape)

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

    def interpolator(self, Z):
        return NotImplemented

    def find_peak(self, Z):
        yield NotImplemented
