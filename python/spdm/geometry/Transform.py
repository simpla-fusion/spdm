from .Solid import Solid
from .Plane import Plane
from .Point import Point
from .Line import Line


class Sweep:
    def __init__(self, shape: Plane | Line, axis: Point | Line, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._section = shape
        self._axis = axis

    @property
    def cross_section(self) -> Plane | Line:
        return self._section

    @property
    def axis(self) -> Point | Line:
        return self._axis


class SweepSurface(Solid):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class SweepSolid(Solid):
    Boundary = SweepSurface

    def __init__(self, shape: Plane, axis:  Line, *args, **kwargs) -> None:
        super().__init__(shape, axis, *args, **kwargs)
        self._boundary = self.__class__.Boundary(shape.boundary, axis, *args)
