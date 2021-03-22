from ..Function import Function


class GeoObject:
    def __init__(self, *args, is_closed=False, **kwargs) -> None:
        self._is_closed = is_closed

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
    def uv(self):
        return NotImplemented

    def point(self, *args, **kwargs):
        return NotImplemented

    def xy(self, *args, **kwargs):
        return self.point(*args, **kwargs).T

    def derivative(self,  *args, **kwargs):
        return NotImplemented

    def pullback(self, func,   *args, **kwargs):
        r"""
            ..math:: f:N\rightarrow M\\\Phi^{*}f:\mathbb{R}\rightarrow M\\\left(\Phi^{*}f\right)\left(u\right)&\equiv f\left(\Phi\left(u\right)\right)=f\left(r\left(u\right),z\left(u\right)\right)
        """
        return func(*self.xy(*args, **kwargs))

    def pullback(self, func,  *args,   **kwargs):
        if len(args) == 0:
            args = self.uv
        return Function(args, func(*self.xy(*args,   **kwargs)), is_period=self.is_closed)
        
    # def dl(self, u, *args, **kwargs):
    #     return NotImplemented
