import collections.abc
from typing import Collection

from spdm.numlib import np

from ..util.logger import logger
from .GeoObject import GeoObject


class Point(GeoObject):
    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 1 and isinstance(args[0], collections.abc.Sequence):
            args = args[0]
        super().__init__(args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._x

    def map(self,  *args, **kwargs):
        return self._x

    def make_one_form(self, *args, **kwargs):
        return lambda *_args: 0.0

    def dl(self, u=None):
        if u is None:
            return np.asarray(0.0)
        else:
            return 0.0*u
