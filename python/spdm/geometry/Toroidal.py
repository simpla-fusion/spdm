
from .Surface import Surface
from .Solid import Solid
from .Plane import Plane
from .Line import Line
from .Circle import Circle


@Surface.register("toroidal_surface")
class ToroidalSurface(Surface):
    def __init__(self, cross_section: Line, circle: Circle, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
            

@Surface.register("toroidal")
class Toroidal(Solid):
    def __init__(self, cross_section: Plane, circle: Circle, *args, **kwargs) -> None:
        super().__init__(**kwargs)
        self._boundary = ToroidalSurface(cross_section.boundary, circle, *args,)
