from functools import cached_property

from ..data.Function import Function
from ..numlib import np
from typing import Sequence


class GeoObject:
    def __init__(self, *args,   **kwargs) -> None:
        pass

    @property
    def topology_rank(self):
        return 0

    @property
    def ndims(self):
        return NotImplemented

    @property
    def uv(self):
        return NotImplemented

    def point(self, *args, **kwargs) -> np.ndarray:
        return NotImplemented

    @cached_property
    def xy(self) -> np.ndarray:
        return self.point().T

    @cached_property
    def is_closed(self):
        return NotImplemented

    @cached_property
    def bbox(self) -> np.ndarray:
        """[[xmin,xmax],[ymin,ymax],...]"""
        return np.asarray([[d.min(), d.max()] for d in self.xy.T])

    @cached_property
    def center(self):
        return (self.bbox[:, 0]+self.bbox[:, 1])*0.5

    def enclosed(self, p: Sequence[float], tolerance=None) -> bool:
        bbox = self.bbox
        if any(p < bbox[:, 0]) or any(p > bbox[:, 1]):
            return False
        elif tolerance is None:
            return True
        else:
            return NotImplemented

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
