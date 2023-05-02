import abc  # abstract base class
import collections.abc
import inspect
import typing
from functools import cached_property

import numpy as np

from spdm.geometry.GeoObject import GeoObject

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
        self._units = kwargs.get("units", ["-"])

    @property
    def type(self) -> str:
        return self._type

    @property
    def units(self) -> typing.List[str]:
        return self._units

    @property
    def geometry(self) -> GeoObject | None:
        """ Geometry of the grid
            网格的几何形状 
        """
        return NotImplemented


@Grid.register([None, 'null'])
class NullGrid(Grid):
    """Null Grid"""
    pass

    @property
    def geometry(self) -> GeoObject | None:
        return None
