from .GeoObject import GeoObject


class Surface(GeoObject):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def topology_rank(self):
        return 2

    @property
    def ndims(self):
        return NotImplemented

    @property
    def points(self):
        return self.map(self._u)

    def point(self, u,  *args, **kwargs):
        return NotImplemented

    def map(self, u, *args, **kwargs):
        return NotImplemented

    def __call__(self, *args, **kwargs):
        return self.map(*args, **kwargs)

    def derivative(self, u, *args, **kwargs):
        return NotImplemented

    def dl(self, u, *args, **kwargs):
        return NotImplemented

    def pullback(self, func, *args, **kwargs):
        return NotImplemented

    def make_one_form(self, func):
        return NotImplemented
