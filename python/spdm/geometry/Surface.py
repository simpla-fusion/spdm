from .GeoObject import GeoObject


class Surface(GeoObject):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, rank=2, **kwargs)
