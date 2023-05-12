import abc
import collections.abc
import typing
from functools import cached_property

import numpy as np
from spdm.geometry.GeoObject import GeoObject

from ..geometry.GeoObject import GeoObject
from ..utils.logger import logger
from ..utils.typing import ArrayType
from .Grid import Grid


class StructuredMesh(Grid):
    def __init__(self,  shape: typing.Sequence[int], ndims=None, cycles=None, **kwargs) -> None:
        shape = tuple(shape)
        ndims = ndims if ndims is not None else len(shape)
        if cycles is None:
            cycles = [False] * ndims
        if not isinstance(cycles, collections.abc.Sequence):
            cycles = [cycles]
        if len(cycles) == 1:
            cycles = cycles * ndims

        super().__init__(shape=shape, ndims=ndims, rank=len(shape), cycles=cycles, **kwargs)

        # logger.debug(f"Create {self.__class__.__name__} rank={self.rank} shape={self.shape} ndims={self.ndims}")

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

    @property
    def points(self) -> typing.Tuple[ArrayType, ...]:
        """ Coordinates of the grid points
            网格点的坐标
            xy.shape == [np.array(shape)]
        """
        return tuple(np.meshgrid(*[np.linspace(0, 1, n) for n in self.shape]))

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
