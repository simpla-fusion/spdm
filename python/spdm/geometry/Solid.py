import typing
import abc
from .GeoObject import GeoObject3D
from .Plane import Plane


class Solid(GeoObject3D):
    """ Line
        线，一维几何体
    """

    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abc.abstractproperty
    def points(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}")

    @abc.abstractproperty
    def boundary(self) -> typing.List[Plane]:
        raise NotImplementedError(f"{self.__class__.__name__}")
