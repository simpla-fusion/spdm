from .Point import Point
import numpy as np
from .GeoObject import GeoObject


class Curve(GeoObject):
    @staticmethod
    def __new__(cls, *args, type=None, **kwargs):
        if len(args) == 0:
            raise RuntimeError(f"Illegal input! {len(args)}")
        shape = [(len(a) if isinstance(a, np.ndarray) else 1) for a in args]
        if all([s == 1 for s in shape]):
            return object.__new__(Point)
        elif cls is not Curve:
            return object.__new__(cls)
        else:
            # FIXME：　find module
            return object.__new__(Curve)

    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def topology_rank(self):
        return 1

    def inside(self, *x):
        return False


class Line(Curve):
    def __init__(self, *args,   **kwargs) -> None:
        super().__init__(*args, is_closed=False, **kwargs)
