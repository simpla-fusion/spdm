import abc
import collections.abc
import typing
from functools import cached_property

import numpy as np
from spdm.geometry.GeoObject import GeoObject

from ..geometry.GeoObject import GeoObject
from ..utils.logger import logger
from .Grid import Grid


class StructuredMesh(Grid):
    def __init__(self,  shape, ndims=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._ndims = ndims if ndims is not None else len(shape)
        self._shape = shape

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

    @property
    def geometry(self) -> GeoObject | None:
        raise NotImplementedError("geometry is not implemented")

    @property
    def cycle(self) -> typing.List[bool]:
        """ Periodic boundary condition
            周期性边界条件
            标识每个维度是否是周期性边界
        """
        return self._cycles

    @property
    def ndims(self) -> int:
        """ Number of dimensions of the space
            空间维度，
        """
        return self._ndims

    @property
    def rank(self) -> int:
        """ Rank of the grid, i.e. number of dimensions
            网格（流形）维度
            rank <=ndims
        """
        return self._rank

    @property
    def shape(self):
        """ Shape of the grid, i.e. number of points in each dimension
            网格上数组的形状         
        """
        return tuple(self._shape)

    # def axis(self, idx, axis=0):
    #     return NotImplemented

    # def remesh(self, *arg, **kwargs):
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

    @cached_property
    def dx(self) -> typing.Sequence[float]:
        """ Grid spacing in each dimension
            每个维度的网格间距
        """
        return NotImplemented

    @property
    def xyz(self) -> typing.List[np.ndarray]:
        """ Coordinates of the grid points
            网格点的坐标
            xy.shape == [np.array(shape)]
        """
        return NotImplemented

    @property
    def uvw(self) -> typing.List[np.ndarray]:
        """
            网格上归一化相对的坐标，取值范围[0,1]
        """
        return np.meshgrid(*[np.linspace(0, 1, n) for n in self.shape])

    def interpolator(self, *args) -> typing.Callable:
        """ Interpolator of the grid
            网格的插值器, 用于网格上的插值
            返回一个函数，该函数的输入是一个坐标，输出是一个值
            输入坐标若为标量，则返回标量值
            输入坐标若为数组，则返回数组
        """
        raise NotImplementedError(args)

    def axis(self, *args, **kwargs) -> GeoObject:
        """ Axis of the grid
            网格的坐标轴，仅仅对于结构化网格有意义
        """
        return NotImplemented

    def axis_iter(self, axis=0) -> typing.Iterator[GeoObject]:
        for idx, u in enumerate(self.uvw[axis]):
            yield u, self.axis(idx, axis=axis)
