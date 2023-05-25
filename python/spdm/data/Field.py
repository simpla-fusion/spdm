from __future__ import annotations

import collections.abc
import typing
from enum import Enum

import numpy as np

from ..mesh.Mesh import Mesh
from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix
from ..utils.tags import _not_found_
from ..utils.typing import ArrayType, NumericType, array_type
from .Function import Function
from .Expression import Expression
from .Profile import Profile
from .Node import Node

_T = typing.TypeVar("_T")


class Field(Profile[_T]):
    """ Field
        ---------
        Field 是 Function 在流形（manifold/Mesh）上的推广， 用于描述流形上的标量场，矢量场，张量场等。

        Field 所在的流形记为 mesh ，可以是任意维度的，可以是任意形状的，可以是任意拓扑的，可以是任意坐标系的。

        Mesh 网格描述流形的几何结构，比如网格的拓扑结构，网格的几何结构，网格的坐标系等。

        Field 与 Function的区别：
            - Function 的 mesh 是 多个一维数组表示dimensions/axis
            - Field 的 mesh 是 Mesh，可以表示复杂流形上的场等。

        当 Field 不做为 DTree 的节点时， 应直接由Function继承  Field(Function[_T])

    """

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._refresh()

    def _refresh(self, force=False) -> Field[_T]:

        if isinstance(self._mesh, Mesh) and not force:
            return

        super()._refresh(force=force)

        if isinstance(self._mesh, collections.abc.Mapping):
            self._mesh = Mesh(**self._mesh)
        elif not isinstance(self._mesh, Mesh):
            logger.warning(f"self._mesh is not a Mesh, but {type(self._mesh)}")

        return self

    @property
    def bbox(self): return self.mesh.geometry.bbox

    @property
    def points(self): return self.mesh.points
    """ 定义域 mesh 上的坐标 """

    @property
    def shape(self) -> typing.Tuple[int]: return self.mesh.shape

    @property
    def ndim(self) -> int: return self.mesh.ndim

    def _compile(self, *d,   **kwargs) -> Field:
        if hasattr(self.__mesh__, "dims"):  # as rectlinear grid
            return super()._compile(*d,  **kwargs)

        # FIMXE: STILL WORKING! NOT FINISH!!!
        
        value = self.__array__()

        if not isinstance(value, array_type):
            raise RuntimeError(f"Function.compile() incorrect value {self.__str__()} value={value}   ")
        elif len(value.shape) == 0 or value.shape == (1,):
            # 如果value是标量，无法插值，则返回 value 作为常函数
            return value

        if tuple(value.shape) != tuple(self.shape):
            raise NotImplementedError(f"{value.shape}!={self.shape}")

        if self.ndim == 1 and len(d) <= 0:
            if hasattr(self.__mesh__, "derivative"):
                return self.__mesh__.derivative(d, value)
            else:
                return super()._compile(*d, **kwargs)
        elif self.ndim == 2 and d == (1,):
            return self._compile(1, 0, **kwargs), self._compile(0, 1, **kwargs)
        elif self.ndim == 3 and d == (1,):
            return self._compile(1, 0, 0, **kwargs), self._compile(0, 1, 0, **kwargs), self._compile(0, 0, 1, **kwargs)
        elif len(d) == 1:
            raise NotImplementedError(f"ndim={self.ndim} d={d}")
        elif len(d) != self.ndim:
            raise RuntimeError(f"Illegal! ndim={self.ndim} d={d}")
        elif all(v == 0 for v in d) and hasattr(self.__mesh__, "interpolator"):
            return self.__mesh__.interpolator(value)
        elif all(v >= 0 for v in d) and hasattr(self.__mesh__, "partial_derivative"):
            return self.__mesh__.partial_derivative(d, value)
        elif all(v <= 0 for v in d) and hasattr(self.__mesh__, "antiderivative"):
            return self.__mesh__.antiderivative([-v for v in d], value)
        else:
            raise NotImplementedError(f"TODO: {d}")

    def grad(self, n=1) -> Field:
        ppoly = self._compile()

        if isinstance(ppoly, tuple):
            ppoly, opts = ppoly
        else:
            opts = {}

        if self.ndim == 2 and n == 1:
            return Field[typing.Tuple[_T, _T]]((ppoly.partial_derivative(1, 0),
                                                ppoly.partial_derivative(0, 1)),
                                               mesh=self.mesh,
                                               name=f"\\nabla({self.__str__()})", **opts)
        elif self.ndim == 3 and n == 1:
            return Field[typing.Tuple[_T, _T, _T]]((ppoly.partial_derivative(1, 0, 0),
                                                    ppoly.partial_derivative(0, 1, 0),
                                                    ppoly.partial_derivative(0, 0, 1)),
                                                   mesh=self.mesh,
                                                   name=f"\\nabla({self.__str__()})", **opts)
        elif self.ndim == 2 and n == 2:
            return Field[typing.Tuple[_T, _T, _T]]((ppoly.partial_derivative(2, 0),
                                                    ppoly.partial_derivative(0, 2),
                                                    ppoly.partial_derivative(1, 1)),
                                                   mesh=self.mesh,
                                                   name=f"\\nabla^{n}({self.__str__()})", **opts)
        else:
            raise NotImplemented(f"TODO: ndim={self.ndim} n={n}")
