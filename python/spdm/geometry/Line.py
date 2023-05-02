import abc
import typing
from functools import cached_property
from typing import Any
import collections.abc
import numpy as np
from spdm.utils.logger import logger

from .GeoObject import GeoObject1D, GeoObject
from .Point import Point
from .Vector import Vector


@GeoObject.register("line")
class Line(GeoObject1D):
    """ Line
        线，一维几何体
    """

    def __init__(self, *args, **kwargs) -> None:
        if self.__class__ is Line:
            from sympy.geometry import Line as _Line
            args = [_Line(*self._normal_points(*args))]

        super().__init__(*args, **kwargs)

    @property
    def is_closed(self) -> bool:
        return False

    @property
    def length(self) -> float:
        return self._geo_entity.length

    @property
    def p0(self) -> Point:
        return Point(self._geo_entity.p1)

    @property
    def p1(self) -> Point:
        return Point(self._geo_entity.p2)

    @property
    def deirection(self) -> Vector:
        return Vector(self._geo_entity.direction[:])

    @property
    def boundary(self) -> typing.List[Point]:
        return np.Infinity

    def contains(self, o) -> bool:
        return self._geo_entity.contains(o._geo_entity if isinstance(o, GeoObject) else o)

 

    def points(self, u: float | np.ndarray | typing.List[float] | None = None) -> np.ndarray | Point | typing.List[Point]:
        p0 = self.p0._geo_entity
        direction = self._geo_entity.direction
        if u is None:
            return tuple([self.p0, self.p1])
        elif isinstance(u, float):
            return (p0 + direction*u)
        elif isinstance(u, np.ndarray):
            return np.asarray([(p0 + direction*v)[:] for v in u])
        elif isinstance(u, collections.abc.Iterable):
            return [Point(p0 + direction*v) for v in u]
        else:
            raise RuntimeError(f"Invalid input type {type(u)}")

    def project(self, o) -> Point:
        return Point(self._geo_entity.projection(o._geo_entity if isinstance(o, GeoObject) else o))

    def bisectors(self, o) -> typing.List[GeoObject]:
        return [GeoObject(v) for v in self._geo_entity.bisectors(o._geo_entity if isinstance(o, GeoObject) else o)]

    @ cached_property
    def dl(self) -> np.ndarray:
        x, y = np.moveaxis(self.points, -1, 0)

        a, b = self.derivative()

        # a = a[:-1]
        # b = b[:-1]
        dx = x[1:]-x[:-1]
        dy = y[1:]-y[:-1]

        m1 = (-a[:-1]*dy+b[:-1]*dx)/(a[:-1]*dx+b[:-1]*dy)

        # a = np.roll(a, 1, axis=0)
        # b = np.roll(b, 1, axis=0)

        m2 = (-a[1:]*dy+b[1:]*dx)/(a[1:]*dx+b[1:]*dy)

        return np.sqrt(dx**2+dy**2)*(1 + (2.0*m1**2+2.0*m2**2-m1*m2)/30)

    def integral(self, func: typing.Callable) -> float:
        x, y = self.xyz
        val = func(x, y)
        # c_pts = self.points((self._mesh[0][1:] + self._mesh[0][:-1])*0.5)

        return np.sum(0.5*(val[:-1]+val[1:]) * self.dl)

    # def average(self, func: Callable[[_TCoord, _TCoord], _TCoord]) -> float:
    #     return self.integral(func)/self.length

    def encloses_point(self, *x: float, **kwargs) -> bool:
        return super().enclosed(**x, **kwargs)

    def trim(self):
        return NotImplemented

    def remesh(self, mesh_type=None):
        return NotImplemented


@ Line.register("ray")
class Ray(Line):
    def __init__(self, *args, **kwargs) -> None:
        if self.__class__ is Ray:
            from sympy.geometry import Ray as _Ray
            args = [_Ray(*self._normal_points(*args))]
        super().__init__(*args, **kwargs)


@ Line.register("segment")
class Segment(Line):
    def __init__(self, *args, **kwargs) -> None:
        if self.__class__ is Ray:
            from sympy.geometry import Segment as _Segment
            args = [_Segment(*self._normal_points(*args))]
        super().__init__(*args, **kwargs)

    @ property
    def midpoint(self) -> Point:
        return Point(self._geo_entity.midpoint)
