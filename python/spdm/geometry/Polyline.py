from .GeoObject import GeoObject1D


class Polyline(GeoObject1D):
    def __init__(self, *args, **kwargs) -> None:
            if self.__class__ is Polyline:
                from sympy.geometry import Curve as _Curve
                args = [_Curve(*self._normal_points(*args))]

            super().__init__(*args, **kwargs)
