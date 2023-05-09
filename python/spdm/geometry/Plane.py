import abc
import typing

import numpy as np

from .GeoObject import GeoObject
from .Line import Line


class Plane(GeoObject):
    """ Plane
        平面，二维几何体
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, rank=2, **kwargs)

    @abc.abstractproperty
    def points(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__}")

    @abc.abstractproperty
    def boundary(self) -> GeoObject:
        raise NotImplementedError(f"{self.__class__.__name__}")

    def map(self, u, *args, **kwargs):
        return NotImplemented

    def derivative(self, u, *args, **kwargs):
        return NotImplemented

    def dl(self, u, *args, **kwargs):
        return NotImplemented

    def pullback(self, func, *args, **kwargs):
        return NotImplemented

    def make_one_form(self, func):
        return NotImplemented
