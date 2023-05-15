import abc


from .GeoObject import GeoObject


class Curve(GeoObject):
    """ Curve
        曲线，一维几何体
    """

    def __init__(self, *args, is_close=False, **kwargs) -> None:
        super().__init__(*args, rank=1, **kwargs)
        self._is_close = is_close
        
    # @abc.abstractproperty
    # def is_convex(self) -> bool: return True

    def is_closed(self) -> bool: return self._is_close
