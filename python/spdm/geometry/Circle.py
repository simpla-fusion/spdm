import typing

from .GeoObject import GeoObject
from .Line import Line
from .Plane import Plane
from .Point import Point
from .Solid import Solid
from .Surface import Surface


@GeoObject.register("circle")
class Circle(GeoObject):
    """ Circle
        圆，具有一个固定圆心和一个固定半径
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, rank=1,  **kwargs)

    @property
    def rank(self) -> int:
        return 1

    @property
    def boundary(self):
        return None

    @property
    def points(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__}")

    @property
    def boundary(self) -> typing.List[Point]:
        raise NotImplementedError(f"{self.__class__.__name__}")

    def map(self, u, *args, **kwargs):
        return NotImplemented

    def derivative(self, u, *args, **kwargs):
        return NotImplemented

    def dl(self, u, *args, **kwargs):
        return NotImplemented

    def pullback(self, func, *args, **kwargs):
        return NotImplemented

    def make_one_form(self, func):
        return NotImplemented


@GeoObject.register("ellipse")
class Ellipse(GeoObject):
    """ Ellipse
        椭圆，具有一个固定圆心和两个固定半径
    """
    pass


@Plane.register("disc")
class Disc(Plane):
    """ Disc
        圆盘
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._boundary = Circle(*args)

    @property
    def boundary(self) -> Circle:
        return self._boundary


@Surface.register("sphere")
class Sphere(Surface):
    """ Sphere
        球面
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._boundary = Circle(*args)

    pass


@Solid.register("ball")
class Ball(GeoObject3D):
    """ Ball
        球体    
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._boundary = Sphere(*args)


@Solid.register("cylinder")
class Cylinder(GeoObject3D):
    """ Cylinder
        圆柱体，具有两个固定端面
    """
    pass


@Surface.register("toroidal_surface")
class ToroidalSurface(Surface):
    def __init__(self, cross_section: Line, circle: Circle, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


@Solid.register("toroidal")
class Toroidal(Solid):
    def __init__(self, section: Plane, circle: Circle, *args, **kwargs) -> None:
        super().__init__(**kwargs)
        self._boundary = ToroidalSurface(cross_section.boundary, circle, *args,)
