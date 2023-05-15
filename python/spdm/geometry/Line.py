import abc
import typing
from functools import cached_property
from typing import Any
import collections.abc
import numpy as np
from ..utils.logger import logger
from ..utils.typing import ArrayType
from .GeoObject import GeoObject, GeoObject
from .Point import Point
from .Vector import Vector


@GeoObject.register("line")
class Line(GeoObject):
    """ Line
        线，一维几何体
    """

    def __init__(self, p0: Point | ArrayType, p1: Point | ArrayType, *args, **kwargs) -> None:
        super().__init__(np.stack([p0, p1]), *args, rank=1,   **kwargs)

    @property
    def p0(self) -> Point: return Point(self._points[0])

    @property
    def p1(self) -> Point: return Point(self._points[1])

    @property
    def is_closed(self) -> bool: return False

    @property
    def length(self) -> float: return self.measure

    @property
    def measure(self) -> float: return np.Inf

    @property
    def direction(self) -> Vector:
        return Vector(self.p1-self.p0)

    @property
    def boundary(self) -> typing.List[Point]:
        return [Point(self.p0), Point(self.p1)]

    def contains(self, o) -> bool:
        return self._impl.contains(o._impl if isinstance(o, GeoObject) else o)

    def coordinates(self, u: float | np.ndarray | typing.List[np.ndarray] | None = None) -> ArrayType:
        direction = self.direction
        if u is None:
            return [self.p0, self.p1]
        elif isinstance(u, float):
            return (self.p0 + self.direction*u)
        elif isinstance(u, np.ndarray):
            return np.asarray([(self.p0 + self.direction*v)[:] for v in u])
        elif isinstance(u, collections.abc.Iterable):
            return np.asarray([(self.p0 + direction*v) for v in u])
        else:
            raise RuntimeError(f"Invalid input type {type(u)}")

    def project(self, o) -> Point:
        return Point(self._impl.projection(o._impl if isinstance(o, GeoObject) else o))

    def bisectors(self, o) -> typing.List[GeoObject]:
        return [GeoObject(v) for v in self._impl.bisectors(o._impl if isinstance(o, GeoObject) else o)]

    @cached_property
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


@Line.register("ray")
class Ray(Line):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


@Line.register("segment")
class Segment(Line):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def midpoint(self) -> Point:
        return Point((self.p0+self.p1)*0.5)
