import abc


from .GeoObject import GeoObject1D


class Curve(GeoObject1D):
    """ Curve
        曲线，一维几何体
    """
    def __init__(self, *args, **kwargs) -> None:
        if self.__class__ is Curve:
            from sympy.geometry import Curve as _Curve
            args = [_Curve(*args)]

        super().__init__(*args, **kwargs)

    @abc.abstractproperty
    def is_convex(self):
        return True

    @abc.abstractproperty
    def is_closed(self):
        return True
