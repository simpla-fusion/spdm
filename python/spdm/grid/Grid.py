import collections.abc
import typing
from functools import cached_property

import numpy as np

from ..geometry.GeoObject import GeoObject
from ..utils.Pluggable import Pluggable


class Grid(Pluggable):

    _plugin_registry = {}

    @classmethod
    def _plugin_guess_name(cls, *args, **kwargs) -> typing.List[str]:
        name_s = kwargs.get("name", None)
        if name_s is None and len(args) > 0 and isinstance(args[0], str):
            name_s = args[0]
        if name_s is None:
            raise ModuleNotFoundError(f"Can find Grid from {args} {kwargs}")

        return [f"spdm.grid.{name_s.capitalize}Grid#{name_s.capitalize}Grid"]

    def __init__(self, *args, **kwargs) -> None:
        if self.__class__ is Grid:
            Pluggable.__init__(self, *args,  **kwargs)
            return
   
    @property
    def name(self) -> str:
        return self._name

    @property
    def unit(self):
        return self._unit

    @property
    def cycle(self):
        return self._cycle

    @property
    def ndims(self) -> int:
        return self._ndims

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def topology_rank(self):
        return self.ndims

    @cached_property
    def bbox(self) -> typing.Sequence[float]:
        return NotImplemented

    @cached_property
    def dx(self) -> typing.Sequence[float]:
        return NotImplemented

    @cached_property
    def boundary(self):
        return NotImplemented

    @property
    def xy(self) -> typing.Sequence[np.ndarray]:
        return NotImplemented

    def new_dataset(self, *args, **kwargs):
        return np.ndarray(self._shape, *args, **kwargs)

    def interpolator(self, Z):
        return NotImplemented

    def axis(self, *args, **kwargs) -> GeoObject:
        return NotImplemented

    def axis_iter(self, axis=0) -> typing.Iterator[GeoObject]:
        for idx, u in enumerate(self._uv[axis]):
            yield u, self.axis(idx, axis=axis)
