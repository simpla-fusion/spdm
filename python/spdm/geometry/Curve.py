import abc


from .GeoObject import GeoObject

class Curve(GeoObject):
    """ Curve
        曲线，一维几何体
    """
    rank = 1

    def __init__(self, *args, **kwargs) -> None:
        if self.__class__ is Curve:
            from sympy.geometry import Curve as _Curve
            args = [_Curve(*args)]

        super().__init__(*args, **kwargs)

    # @abc.abstractproperty
    # def is_convex(self):
    #     return True

    # @abc.abstractproperty
    # def is_closed(self):
    #     return True
