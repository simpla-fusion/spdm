from ..util.logger import logger
from .GeoObject import GeoObject
from spdm.numlib import np


class Point(GeoObject):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._x = args

    @property
    def topology_rank(self):
        return 0

    @property
    def ndims(self):
        return len(self._x)

    def __call__(self, *args, **kwargs):
        return self._x

    def map(self,  *args, **kwargs):
        return self._x

    @property
    def points(self):
        return self._x

    def make_one_form(self, *args, **kwargs):
        return lambda *_args: 0.0

    def dl(self, u=None):
        if u is None:
            return np.asarray(0.0)
        else:
            return 0.0*u
