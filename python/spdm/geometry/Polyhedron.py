import abc
import typing
import numpy as np
from .Point import Point
from .Line import Line, Segment
from .Plane import Plane
from .GeoObject import GeoObject3D, GeoObjectSet
from .Polygon import Polygon


class Polyhedron(GeoObject3D):

    @property
    def is_convex(self) -> bool:
        return True

    @property
    def vertices(self) -> typing.Generator[Point, None, None]:
        raise NotImplementedError()

    @property
    def edges(self) -> typing.Generator[Segment, None, None]:
        raise NotImplementedError()

    @property
    def faces(self) -> typing.Generator[Polygon, None, None]:
        raise NotImplementedError()

    @property
    def boundary(self) -> typing.Generator[Polygon, None, None]:
        yield from self.faces
