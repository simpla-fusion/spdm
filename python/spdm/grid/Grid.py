import collections.abc
import typing
from functools import cached_property
import inspect
import numpy as np

from ..geometry.GeoObject import GeoObject
from ..utils.Pluggable import Pluggable


class Grid(Pluggable):

    _plugin_registry = {}

    @classmethod
    def __dispatch__init__(cls, _grid_type, self, *args, **kwargs) -> None:
        if _grid_type is None or len(_grid_type) == 0:
            _grid_type = kwargs.get("grid_type", None)
            if _grid_type is None and len(args) > 0 and isinstance(args[0], str):
                _grid_type = args[0]

            kwargs["grid_type"] = _grid_type

            if isinstance(_grid_type, str):
                _grid_type = [_grid_type,
                             f"spdm.grid.{_grid_type}Grid#{_grid_type}Grid",
                             f"spdm.grid.{_grid_type}Mesh#{_grid_type}Mesh"
                             f"spdm.grid.{_grid_type.capitalize()}Grid#{_grid_type.capitalize()}Grid",
                             f"spdm.grid.{_grid_type.capitalize()}Mesh#{_grid_type.capitalize()}Mesh"
                             
                             ]
            else:
                _grid_type = [_grid_type]

        super().__dispatch__init__(_grid_type, self, *args, **kwargs)

    def __init__(self, *args, **kwargs) -> None:
        if self.__class__ is Grid:
            return Grid.__dispatch__init__(None, self, *args, **kwargs)
        self._type = kwargs.get("grid_type", "unnamed")
        self._units = kwargs.get("units", [])
        self._cycles = kwargs.get("cycles", [])

    @property
    def type(self) -> str:
        return self._type

    @property
    def units(self) -> list:
        return self._units

    @property
    def cycle(self):
        return self._cycles

    @property
    def ndims(self) -> int:
        return self._ndims

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def shape(self):
        return tuple(self._shape)

    def get_shape(self, *args) -> typing.List[int]:
        return NotImplemented

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

    def interpolator(self, *args) -> typing.Callable:
        raise NotImplementedError(args)

    def axis(self, *args, **kwargs) -> GeoObject:
        return NotImplemented

    def axis_iter(self, axis=0) -> typing.Iterator[GeoObject]:
        for idx, u in enumerate(self._uv[axis]):
            yield u, self.axis(idx, axis=axis)


@Grid.register()
class NullGrid(Grid):
    """Null Grid"""
    pass
