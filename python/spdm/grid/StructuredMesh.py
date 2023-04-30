import collections.abc

from ..utils.logger import logger
from .Grid import Grid


class StructuredMesh(Grid):
    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # ndims = None, uv = None, rank = None, shape = None, name = None, unit = None, cycle = None,

        self._shape = kwargs.get('shape', [])
        self._rank = kwargs.get('rank', None) or len(self._shape)
        self._ndims: int = kwargs.get('rank',  self._rank)
        self._uv = args

        # name = name or [""] * self._ndims
        # if isinstance(name, str):
        #     self._name = name.split(",")
        # elif not isinstance(name, collections.abc.Sequence):
        #     self._name = [name]
        # unit = unit or [None] * self._ndims
        # if isinstance(unit, str):
        #     unit = unit.split(",")
        # elif not isinstance(unit, collections.abc.Sequence):
        #     unit = [unit]
        # if len(unit) == 1:
        #     unit = unit * self._ndims
        # # self._unit = [*map(Unit(u for u in unit))]

        cycle = kwargs.get('cycle', None)
        if cycle is None:
            cycle = [False] * self._ndims
        if not isinstance(cycle, collections.abc.Sequence):
            cycle = [cycle]
        if len(cycle) == 1:
            cycle = cycle * self._ndims
        self._cycle = cycle

        # logger.debug(f"Create {self.__class__.__name__} rank={self.rank} shape={self.shape} ndims={self.ndims}")

    # def axis(self, idx, axis=0):
    #     return NotImplemented

    # def remesh(self, *arg, **kwargs):
    #     return NotImplemented

    # def interpolator(self, Z):
    #     return NotImplemented

    # def find_critical_points(self, Z):
    #     X, Y = self.points
    #     xmin = X.min()
    #     xmax = X.max()
    #     ymin = Y.min()
    #     ymax = Y.max()
    #     dx = (xmax-xmin)/X.shape[0]
    #     dy = (ymax-ymin)/X.shape[1]
    #     yield from find_critical_points(self.interpolator(Z),  xmin, ymin, xmax, ymax, tolerance=[dx, dy])

    # def sub_axis(self, axis=0) -> Iterator[GeoObject]:
    #     for idx in range(self.shape[axis]):
    #         yield self.axis(idx, axis=axis)
