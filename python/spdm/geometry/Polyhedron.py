import abc
import typing

import numpy as np
from ..utils.typing import ArrayType
from .GeoObject import GeoObject
from .Line import Line, Segment
from .Plane import Plane
from .Point import Point
from .Polygon import Polygon
from .PointSet import PointSet


class Polyhedron(PointSet):

    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args, rank=3, **kwargs)

    @property
    def is_convex(self) -> bool: return True

    @property
    def edges(self) -> typing.Generator[Segment, None, None]:
        raise NotImplementedError()

    @property
    def faces(self) -> typing.Generator[Polygon, None, None]:
        raise NotImplementedError()

    @property
    def boundary(self) -> typing.Generator[Polygon, None, None]:
        yield from self.faces
