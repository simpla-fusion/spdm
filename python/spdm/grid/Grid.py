from __future__ import annotations

import collections.abc
import typing
from functools import cached_property

from ..geometry.GeoObject import GeoObject, GeoObjectSet, as_geo_object
from ..utils.logger import logger
from ..utils.misc import regroup_dict_by_prefix
from ..utils.Pluggable import Pluggable
from ..utils.typing import ArrayType, NumericType, ScalarType


class Grid(Pluggable):

    _plugin_registry = {}

    @classmethod
    def __dispatch__init__(cls, _grid_type, self, *args, **kwargs) -> None:
        if not _grid_type:
            _grid_type = kwargs.get("type", RegularGrid)

        if isinstance(_grid_type, str):
            _grid_type = [_grid_type,
                          f"spdm.grid.{_grid_type}Grid#{_grid_type}Grid",
                          f"spdm.grid.{_grid_type}Mesh#{_grid_type}Mesh",
                          f"spdm.grid.{_grid_type.capitalize()}Grid#{_grid_type.capitalize()}Grid",
                          f"spdm.grid.{_grid_type.capitalize()}Mesh#{_grid_type.capitalize()}Mesh"
                          ]

        super().__dispatch__init__(_grid_type, self, *args, **kwargs)

    def __init__(self, *args, **kwargs) -> None:
        if self.__class__ is Grid:
            return Grid.__dispatch__init__(None, self, *args, **kwargs)

        self._geometry, self._metadata = regroup_dict_by_prefix(kwargs, "geometry")

        if isinstance(self._geometry, collections.abc.Mapping) or self._geometry is None:
            self._geometry = as_geo_object(*args, **self._geometry)

        if not isinstance(self._geometry, (GeoObject, GeoObjectSet)):
            raise ValueError(f"Grid.__init__(): geometry={self._geometry} is not a GeoObject or GeoObjectSet")

        self._shape: typing.Tuple[int] = self._metadata.get("shape", None)

        self._cycles: typing.Tuple[int] = self._metadata.get("cycles", None) or ([False]*self.geometry.rank)

        if len(args) > 0:
            raise RuntimeWarning(f"Grid.__init__(): {args} are ignored")

    def __serialize__(self) -> typing.Mapping:
        raise NotImplementedError(f"")

    @classmethod
    def __deserialize__(cls, data: typing.Mapping) -> Grid:
        raise NotImplementedError(f"")

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def name(self) -> str: return self._metadata.get("name", 'unamed')

    @property
    def type(self) -> str: return self._metadata.get("type", "regular")

    @property
    def units(self) -> typing.Tuple[str, ...]: return tuple(self._metadata.get("units", ["-"]))

    @property
    def geometry(self) -> GeoObject | GeoObjectSet: return self._geometry
    """ Geometry of the grid  网格的几何形状  """

    @property
    def shape(self) -> typing.Tuple[int, ...]: return self._shape
    """ 存储网格点数组的形状  TODO: support multiblock grid"""

    @property
    def cycles(self) -> typing.List[bool]: return self._cycles
    """ Periodic boundary condition   周期性边界条件,  标识每个维度是否是周期性边界 """

    @property
    def uv_points(self) -> typing.Tuple[ArrayType, ...]: return self._uv_points

    @property
    def points(self) -> typing.Tuple[ArrayType, ...]:
        """ 网格点坐标 """
        if self._geometry is None:
            return self.uv_points
        else:
            logger.debug(self.uv_points)
            res = self._geometry.points(*self.uv_points)
            logger.debug(res.shape)
            return res

    def interpolator(self, y: NumericType, *args, **kwargs) -> typing.Callable[..., NumericType]:
        raise NotImplementedError(f"{self.__class__.__name__}.interpolator")

    def partial_derivative(self, y: NumericType, *args, **kwargs) -> typing.Callable[..., NumericType]:
        raise NotImplementedError(f"{self.__class__.__name__}.partial_derivative")

    def antiderivative(self, y:  NumericType, *args, **kwargs) -> typing.Callable[..., NumericType]:
        raise NotImplementedError(f"{self.__class__.__name__}.antiderivative")

    def integrate(self, y:  NumericType, *args, **kwargs) -> ScalarType:
        raise NotImplementedError(f"{self.__class__.__name__}.integrate")


@Grid.register("regular")
class RegularGrid(Grid):
    pass


def as_grid(*args, **kwargs) -> Grid:
    if len(args) == 1 and isinstance(args[0], Grid):
        if len(kwargs) > 0:
            logger.warning(f"Ignore kwargs {kwargs}")
        return args[0]
    else:
        return Grid(*args, **kwargs)
