from __future__ import annotations

import collections.abc
import typing
from functools import cached_property
import abc
import numpy as np


from ..utils.logger import logger
from ..utils.Pluggable import Pluggable


class GeoObject(Pluggable):
    """ Geomertic object
    几何对象，包括点、线、面、体等

    NOTE: 基于sympy.geometry 实现
    """

    _plugin_registry = {}

    @classmethod
    def __dispatch__init__(cls, _geo_type, self, *args, **kwargs) -> None:
        """
        """
        if _geo_type is None or len(_geo_type) == 0:
            _geo_type = kwargs.get("geometry_type", None)

            if _geo_type is None and len(args) > 0 and isinstance(args[0], str):
                _geo_type = args[0]
                args = args[1:]

            if isinstance(_geo_type, str):
                _geo_type = [_geo_type,
                             f"spdm.geometry.{_geo_type}#{_geo_type}",
                             f"spdm.geometry.{_geo_type}{cls.__name__}#{_geo_type}{cls.__name__}",
                             f"spdm.geometry.{_geo_type.capitalize()}#{_geo_type.capitalize()}",
                             f"spdm.geometry.{_geo_type.capitalize()}{cls.__name__}#{_geo_type.capitalize()}{cls.__name__}",
                             f"spdm.geometry.{cls.__name__}#{_geo_type}"
                             ]
            else:
                _geo_type = [_geo_type]

            kwargs["geometry_type"] = _geo_type

        super().__dispatch__init__(_geo_type, self, *args, **kwargs)

    def __init__(self, *args,  **kwargs) -> None:
        if self.__class__ is GeoObject:
            return GeoObject.__dispatch__init__(None, self, *args, **kwargs)

        from sympy.geometry.entity import GeometryEntity

        if len(args) == 0:
            pass
        elif isinstance(args[0], GeometryEntity):
            self._geo_entity = args[0]
            args = args[1:]
            self._appinfo = kwargs
            if len(args) > 0:
                logger.warning(f"Ignore extra arguments: {args[1:]}")
        elif self.__class__ is GeoObject:
            return GeoObject.__dispatch__init__(None, self, *args, **kwargs)
        else:
            raise TypeError("The first argument must be a GeometryEntity")

    def __equal__(self, other: GeoObject) -> bool:
        return isinstance(other, GeoObject) and self._geo_entity == other._geo_entity

    def _repr_svg_(self) -> str:
        return self._geo_entity._repr_svg_()

    def __svg__(self) -> str:
        return self._geo_entity._svg()

    def __getitem__(self, *args) -> typing.Any:
        return self._geo_entity.__getitem__(*args)

    @property
    def ndims(self) -> int:
        return self._geo_entity.ambient_dimension

    @property
    def rank(self) -> int:
        """
            0: point
            1: curve
            2: surface
            3: volume
            >=4: not defined
        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    @property
    def bounds(self) -> typing.Tuple[float]:
        return self._geo_entity.bounds

    @property
    def is_convex(self) -> bool:
        return self._geo_entity.is_convex()

    @property
    def center(self):
        return (np.array(self.bounds[::2])+np.array(self.bounds[1::2]))*0.5

    def enclose(self, other) -> bool:
        """ Return True if all args are inside the geometry, False otherwise.
        """
        return self._geo_entity.encloses(GeoObject(other)._geo_entity)

    def intersection(self, other) -> typing.Set[GeoObject]:
        """ Return the intersection of self with other. """
        return {GeoObject(o) for o in self._geo_entity.intersection(GeoObject(other)._geo_entity)}

    def reflect(self, line) -> GeoObject:
        """ reflect self by line"""
        return GeoObject(self._geo_entity.reflect(GeoObject(line)._geo_entity))

    def rotate(self, angle, pt=None) -> GeoObject:
        return GeoObject(self._geo_entity.rotate(angle, GeoObject(pt)._geo_entity if pt is not None else None))

    def scale(self, x=1, y=1, pt=None) -> GeoObject:
        """ scale self by x, y, pt
        """
        return GeoObject(self._geo_entity.scale(x, y, GeoObject(pt)._geo_entity if pt is not None else None))

    def translate(self, *args) -> GeoObject:
        return GeoObject(self._geo_entity.translate(*args))

    def __call__(self, *args: float | np.ndarray, **kwargs) -> np.ndarray | typing.List[float]:
        res = self.points(*args, **kwargs)
        if not isinstance(res, np.ndarray):
            res = res[:]
        return res

    def points(self, *uv, **kwargs) -> np.ndarray:
        """
            coordinates of vertices on mesh
            Return: array-like
                shape = [*shape_of_mesh,ndims]
        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    @property
    def xyz(self) -> np.ndarray:
        """
            coordinates of vertices on mesh
            Return: array-like
                shape = [ndims, *shape_of_mesh]
        """
        return np.moveaxis(self.points(), -1, 0)

    def dl(self, uv=None) -> np.ndarray:
        """
            derivative of shape
            Returns:
                rank==0 : 0
                rank==1 : dl (shape=[n-1])
                rank==2 : dx (shape=[n-1,m-1]), dy (shape=[n-1,m-1])
                rank==3 : dx (shape=[n-1,m-1,l-1]), dy (shape=[n-1,m-1,l-1]), dz (shape=[n-1,m-1,l-1])
        """
        return np.asarray(0)

    def integral(self, func: typing.Callable) -> float:
        return NotImplemented

    # def average(self, func: typing.Callable[[_TCoord, _TCoord], _TCoord]) -> float:
    #     return self.integral(func)/self.length

    @cached_property
    def is_closed(self):
        if self.rank == 0:
            return True
        else:
            return np.allclose(self.xyz[:, 0], self.xyz[:, -1])

    def trim(self):
        return NotImplemented

    def remesh(self, mesh_type=None, /, **kwargs):
        return NotImplemented

    def derivative(self,  *args, **kwargs):
        return NotImplemented

    def pullback(self, func,   *args, **kwargs):
        r"""
            ..math:: f:N\rightarrow M\\\Phi^{*}f:\mathbb{R}\rightarrow M\\\left(\Phi^{*}f\right)\left(u\right)&\equiv f\left(\Phi\left(u\right)\right)=f\left(r\left(u\right),z\left(u\right)\right)
        """
        # if len(args) == 0:
        #     args = self._mesh

        # return Function(args, func(*self.xyz(*args,   **kwargs)), is_period=self.is_closed)
        return NotImplemented

    @classmethod
    def _normal_points(cls, *args):
        from sympy.geometry import Point as _Point
        if len(args) == 0:
            return []
        elif len(args) == 1:
            if isinstance(args[0], _Point):
                return args[0]
            elif isinstance(args[0], collections.abc.Sequence):
                return _Point(*args[0])
            else:
                return args[0]
        elif isinstance(args[0], (int, float, bool, complex)):
            return _Point(*args)
        elif isinstance(args[0], collections.abc.Sequence):
            return (cls._normal_points(*p) for p in args)
        else:
            raise TypeError(f"args has wrong type {type(args[0])} {args}")


class GeoObject0D(GeoObject):
    @property
    def rank(self) -> int:
        return 0

    def points(self) -> GeoObject:
        raise NotImplementedError(f"{self.__class__.__name__}")


class GeoObject1D(GeoObject):
    @property
    def rank(self) -> int:
        return 1

    @abc.abstractproperty
    def boundary(self) -> GeoObject0D:
        return NotImplemented


class GeoObject2D(GeoObject):
    @property
    def rank(self) -> int:
        return 2

    @abc.abstractproperty
    def boundary(self) -> GeoObject1D:
        return NotImplemented


class GeoObject3D(GeoObject):
    @property
    def rank(self) -> int:
        return 3

    @abc.abstractproperty
    def boundary(self) -> GeoObject2D:
        return NotImplemented
