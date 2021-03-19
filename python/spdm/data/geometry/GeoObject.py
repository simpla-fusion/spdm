

class GeoObject:
    def __init__(self, *args, is_closed=False, **kwargs) -> None:
        self._is_closed = is_closed
        self._uv = None

    @property
    def topology_rank(self):
        return 0

    @property
    def is_closed(self):
        return self._is_closed

    @property
    def ndims(self):
        return NotImplemented

    @property
    def points(self):
        return self.map(self._u)

    def inside(self, *x):
        return False

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
        r"""
            ..math:: f:N\rightarrow M\\\Phi^{*}f:\mathbb{R}\rightarrow M\\\left(\Phi^{*}f\right)\left(u\right)&\equiv f\left(\Phi\left(u\right)\right)=f\left(r\left(u\right),z\left(u\right)\right)
        """
        return func(*self.map(*args, **kwargs))

    def make_one_form(self, func):
        return NotImplemented
