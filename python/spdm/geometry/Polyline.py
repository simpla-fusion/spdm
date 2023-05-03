from .GeoObject import GeoObject1D, GeoObjectSet
from .Line import Segment
from .Point import Point


class Polyline(GeoObject1D):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
