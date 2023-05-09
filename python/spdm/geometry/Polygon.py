import collections.abc
import typing

from .GeoObject import GeoObject
from .Line import Segment
from .Point import Point
from .Polyline import Polyline


class Polygon(GeoObject):
    """ Polygon
        多边形

    """

    _dispatch__init__ = None

    def __init__(self, *args, **kwargs) -> None:
        if len(args) >= 3:
            if Polygon.__dispatch__init__ is None:
                raise RuntimeError(f"Polygon.__dispatch__init__ is None")
            return Polygon.__dispatch__init__(self, *args, **kwargs)
        super().__init__(*args, rank=2, **kwargs)

    @property
    def vertices(self) -> typing.Generator[Point, None, None]:
        for p in self._impl.vertices:
            yield Point(p)

    @property
    def edges(self) -> typing.Generator[Segment, None, None]:
        pt_iter = self.vertices
        first_pt = next(pt_iter)
        current_pt = first_pt
        while True:
            try:
                next_pt = next(pt_iter)
                yield Segment(current_pt, next_pt)
                current_pt = next_pt
            except StopIteration:
                yield Segment(current_pt, first_pt)
                break

    @property
    def boundary(self) -> Polyline:
        return Polyline([*self.vertices], is_closed=True)

    def points(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}")


@Polygon.register("regular_polygon")
class RegularPolygon(Polygon):
    """ Regular Polygon
        正多边形

        cneter: Point or tuple or list or sympy.Point 
            中心点
        radius: float
            半径
        num_of_edges: int
            边数
        rot: float
            旋转角度
    """

    def __init__(self, center: Point, radius: float, num_of_edges: int, *args, **kwargs) -> None:
        center = Point(center) if not isinstance(center, Point) else center
        from sympy.geometry.polygon import RegularPolygon as _RegularPolygon
        args = [_RegularPolygon(center._impl, radius, num_of_edges, *args)]
        super().__init__(*args, **kwargs)

    @property
    def area(self) -> float:
        return self._impl.area

    @property
    def length(self) -> float:
        return self._impl.length

    @property
    def center(self) -> Point:
        return Point(self._impl.center)

    @property
    def radius(self) -> float:
        return self._impl.radius

    @property
    def inradius(self) -> float:
        return self._impl.inradius

    @property
    def rotation(self) -> float:
        return self._impl.rotation

    @property
    def vertices(self) -> typing.Generator[Point, None, None]:
        for p in self._impl.vertices:
            yield Point(p)


@Polygon.register("triangle")
class Triangle(Polygon):
    pass


@Polygon.register("rectangle")
class Rectangle(Polygon):
    pass


@Polygon.register("pentagon")
class Pentagon(Polygon):
    pass


@Polygon.register("hexagon")
class Hexagon(Polygon):
    pass


def _polygon__dispatch__init__(self, *args, **kwargs):
    args = GeoObject._normal_points(*args)
    from sympy.geometry import Polygon as _Polygon
    if not isinstance(args, collections.abc.Sequence):
        return
    match len(args):
        case 3:
            from sympy.geometry.polygon import Triangle as _Triangle
            self.__class__ = Triangle
            return Triangle.__init__(self, _Triangle(*args), **kwargs)
        case 4:
            self.__class__ = Rectangle
            return Rectangle.__init__(self, _Polygon(*args), **kwargs)
        case 5:
            self.__class__ = Pentagon
            return Pentagon.__init__(self, _Polygon(*args), **kwargs)
        case 6:
            self.__class__ = Hexagon
            return Hexagon.__init__(self, _Polygon(*args), **kwargs)
        case _:

            return Polygon.__init__(self, _Polygon(*args), **kwargs)


Polygon.__dispatch__init__ = _polygon__dispatch__init__
