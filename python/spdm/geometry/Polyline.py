from .GeoObject import GeoObject
from .Line import Segment
from .Point import Point


class Polyline(GeoObject):

    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args, rank=1, **kwargs)
