from typing import Callable

from ..numerical.optimize import find_critical_points
from ..util.logger import logger
from .Mesh import Mesh


class StructuredMesh(Mesh):
    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def axis(self, idx, axis=0):
        return NotImplemented

    def remesh(self, *arg, **kwargs):
        return NotImplemented

    def interpolator(self, Z):
        return NotImplemented

    def find_critical_points(self, Z):
        X, Y = self.points
        yield from find_critical_points(self.interpolator(Z), X, Y)
