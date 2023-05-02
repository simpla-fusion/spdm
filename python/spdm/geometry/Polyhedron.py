import abc
import typing
import numpy as np
from .Point import Point
from .Line import Line, Segment
from .Plane import Plane
from .GeoObject import GeoObject3D, GeoObject2D
from .Polygon import Polygon


class Polyhedron(GeoObject3D):

    @property
    def is_convex(self) -> bool:
        return True

    @property
    def vertices(self) -> typing.Set[Point] | np.ndarray:
        raise NotImplementedError()

    @property
    def edges(self) -> typing.Set[Segment]:
        raise NotImplementedError()

    @property
    def faces(self) -> typing.Set[Polygon]:
        raise NotImplementedError()

    @abc.abstractproperty
    def boundary(self) -> typing.Set[Plane]:
        return self._boundary
