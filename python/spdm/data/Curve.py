from .Point import Point
import numpy as np


class Curve:

    @staticmethod
    def __new__(cls, *args, type=None, **kwargs):
        if len(args) == 0:
            raise RuntimeError(f"Illegal input! {len(args)}")
        shape = [(len(a) if isinstance(a, np.ndarray) else 1) for a in args]
        if  all([s == 1 for s in shape]):
            return object.__new__(Point)
        elif cls is not Curve:
            return object.__new__(cls)
        else:
            # FIXME：　find module
            return object.__new__(Curve)

    def __init__(self, *args, is_closed=False, **kwargs) -> None:
        self._is_closed = is_closed

    @property
    def is_closed(self):
        return self._is_closed

    def inside(self, *x):
        return False
