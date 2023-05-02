import typing
import numpy as np
from .Point import Point


class Vector(np.ndarray):
    def __init__(self, *args):
        super().__init__(list(args))
