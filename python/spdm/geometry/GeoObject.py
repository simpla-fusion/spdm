from __future__ import annotations

import collections.abc
import typing
from copy import copy
import uuid
from functools import cached_property

import numpy as np

from ..data.List import List
from ..utils.logger import logger
from ..utils.Pluggable import Pluggable
from ..utils.typing import (ArrayLike, ArrayType, NumericType, ScalarType,
                            array_type, nTupleType, numeric_type)


class BBox:
    def __init__(self, xmin: ArrayType, xmax: ArrayType) -> None:
        self._xmin = xmin
        self._xmax = xmax

    def __equal__(self, other: BBox) -> bool:
        return np.isclose(self._xmin == other._xmin) and np.isclose(self._xmax == other._xmax)

    def __iter__(self) -> typing.Generator[ArrayType, None, None]:
        yield self._xmin
        yield self._xmax

    @property
    def ndim(self) -> int: return len(self._xmin)

    @property
    def center(self) -> ArrayType: return (self._xmin+self._xmax)*0.5
    """ center of geometry """

    @property
    def measure(self) -> ScalarType: return np.product(self._xmax-self._xmin)
    """ measure of geometry, length,area,volume,etc. 默认为 bbox 的体积 """

    def enclose(self, *args) -> bool:
        """ Return True if all args are inside the geometry, False otherwise. """

        if len(args) == 1 and isinstance(args[0], BBox):
            other: BBox = args[0]
            return np.all(self._xmin <= other._xmin) and np.all(self._xmax >= other._xmax)
        else:
            other = np.asarray(args)
            return np.all(self._xmin <= other) and np.all(self._xmax >= other)

    def union(self, other: BBox) -> BBox: raise NotImplementedError(f"intersection")
    """ Return the union of self with other. """

    def intersection(self, other: BBox): raise NotImplementedError(f"intersection")
    """ Return the intersection of self with other. """

    def reflect(self, point0, pointt1): raise NotImplementedError(f"reflect")
    """ reflect  by line"""

    def rotate(self, angle, axis=None): raise NotImplementedError(f"rotate")
    """ rotate  by angle and axis"""

    def scale(self, *s, point=None): raise NotImplementedError(f"scale")
    """ scale self by *s, point """

    def translate(self, *shift): raise NotImplementedError(f"translate")


class GeoObject(Pluggable):
    """ Geomertic object
    几何对象，包括点、线、面、体等

    TODO:
        - 目前基于sympy.geometry实现，未来将支持其他几何建模库
        - 支持3D可视化 （Jupyter+？）

    """
    _plugin_registry = {}

    @classmethod
    def __dispatch__init__(cls, _geo_type, self, *args, **kwargs) -> None:
        """
        """
        if _geo_type is None or len(_geo_type) == 0:
            _geo_type = kwargs.pop("type", Box)

        if isinstance(_geo_type, str):
            _geo_type = [_geo_type,
                         f"spdm.geometry.{_geo_type}#{_geo_type}",
                         f"spdm.geometry.{_geo_type}{cls.__name__}#{_geo_type}{cls.__name__}",
                         f"spdm.geometry.{_geo_type.capitalize()}#{_geo_type.capitalize()}",
                         f"spdm.geometry.{_geo_type.capitalize()}{cls.__name__}#{_geo_type.capitalize()}{cls.__name__}",
                         f"spdm.geometry.{cls.__name__}#{_geo_type}"
                         ]

        super().__dispatch__init__(_geo_type, self, *args, **kwargs)

    def __init__(self, *args, ndim: int = 0, rank: int = 0,  **kwargs) -> None:
        if self.__class__ is GeoObject:
            return GeoObject.__dispatch__init__(None, self, *args, **kwargs)

        self._name = kwargs.pop("name", None)
        self._metadata = kwargs
        self._rank = rank
        self._ndim = ndim

    def __copy__(self) -> GeoObject:
        other: GeoObject = self.__class__(rank=self.rank, ndim=self.ndim)
        other._name = f"{self._name}_copy"
        other._metadata = self._metadata
        return other

    def __svg__(self) -> str:
        if self.ndim != 2:
            raise NotImplementedError(f"{self.__class__.__name__}.__svg__ ndim={self.ndim}")
        else:
            xmin = self.bbox._xmin
            xmax = self.bbox._xmax
            if self.rank == 0:
                return f"<circle cx='{xmin[0]}' cy='{xmin[1]}' r='3' stroke='black' stroke-width='1' fill='red' />"
            else:
                return f"<rect x='{xmin[0]}' y='{xmin[1]}' width='{xmax[0]-xmin[0]}' height='{xmax[1]-xmin[1]}' stroke='black' stroke-width='1' fill='none' />"

    def _repr_svg_(self) -> str:
        xmin, xmax = self.bbox
        width = xmax[0]-xmin[0]
        height = xmax[1]-xmin[1]

        scale = 1.0
        shift_x = xmin[0]
        shift_y = xmin[1]

        return f"""<svg width={width} height={height}'>
                    <g id={self.name} transform='scale({scale}),shift({shift_x},{shift_y})' >
                        {self.__svg__()}
                    </g>
                   </svg>"""

    def _repr_html_(self) -> str: return self._repr_svg_()

    def __equal__(self, other: GeoObject) -> bool:
        return isinstance(other, GeoObject) and self.rank == other.rank and self.ndim == other.ndim and self.bbox == other.bbox

    @property
    def name(self) -> str:
        if self._name is None:
            self._name = f"{self.__class__.__name__}_{uuid.uuid1()}"
        return self._name

    @property
    def rank(self) -> int: return self._rank
    """ 几何体（流形）维度  rank <=ndims

            0: point
            1: curve
            2: surface
            3: volume
            >=4: not defined
        The rank of a geometric object refers to the number of independent directions
        in which it extends. For example, a point has rank 0, a line has rank 1,
        a plane has rank 2, and a volume has rank 3.
    """

    @property
    def dimension(self) -> int: return self._ndim
    """ 几何体所处的空间维度， = 0，1，2，3 ,...
        The dimension of a geometric object, on the other hand, refers to the minimum number of
        coordinates needed to specify any point within it. In general, the rank and dimension of
        a geometric object are the same. However, there are some cases where they can differ.
        For example, a curve that is embedded in three-dimensional space has rank 1 because
        it extends in only one independent direction, but it has dimension 3 because three
        coordinates are needed to specify any point on the curve.
    """

    @property
    def ndim(self) -> int: return self._ndim
    """ alias of dimension """

    @property
    def is_convex(self) -> bool: return True
    """ is convex """

    @property
    def is_closed(self): return False

    @property
    def bbox(self) -> BBox: raise NotImplementedError(f"{self.__class__.__name__}.bbox")
    """ bbox of geometry """

    @property
    def center(self) -> GeoObject: return as_geo_object(self.bbox.center)
    """ center of geometry """

    @property
    def measure(self) -> ScalarType: return self.bbox.measure
    """ measure of geometry, length,area,volume,etc. 默认为 bbox 的体积 """

    def enclose(self, other: GeoObject) -> bool: return self.bbox.enclose(other.bbox)
    """ Return True if all args are inside the geometry, False otherwise. """

    def intersection(self, other: GeoObject) -> typing.List[GeoObject]:
        """ Return the intersection of self with other. """
        return [as_geo_object(self.bbox.intersection(other.bbox))]

    def reflect(self, point0, point1) -> GeoObject:
        """ reflect  by line"""
        other = copy(self)
        other._name = f"{self.name}_reflect"
        other.bbox.reflect(point0, point1)
        return other

    def rotate(self, angle, axis=None) -> GeoObject:
        """ rotate  by angle and axis"""
        other = copy(self)
        other._name = f"{self.name}_rotate"
        other.bbox.rotate(angle, axis=axis)
        return other

    def scale(self, *s, point=None) -> GeoObject:
        """ scale self by *s, point """
        other = copy(self)
        other._name = f"{self.name}_scale"
        other.bbox.scale(*s, point=point)
        return other

    def translate(self, *shift) -> GeoObject:
        other = copy(self)
        other._name = f"{self.name}_translate"
        other.bbox.translate(*shift)
        return other

    @property
    def boundary(self) -> typing.List[GeoObject]: raise NotImplementedError(f"{self.__class__.__name__}.boundary")
    """ boundary of geometry which is a geometry of rank-1 """

    def trim(self): raise NotImplementedError(f"{self.__class__.__name__}.trim")

    @staticmethod
    def _normal_points(*args) -> np.ndarray | typing.List[float]:
        if len(args) == 0:
            return []
        elif len(args) == 1:
            return args[0]
        elif isinstance(args[0], (int, float, bool, complex)):
            return list(args)
        elif isinstance(args[0], collections.abc.Sequence):
            return np.asarray([GeoObject._normal_points(*p) for p in args])
        else:
            raise TypeError(f"args has wrong type {type(args[0])} {args}")


class Box(GeoObject):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def bbox(self) -> typing.Tuple[ArrayType, ArrayType]: return self._points[0], self._points[1]

    def enclose(self, *xargs) -> bool:
        if all([isinstance(x, numeric_type) for x in xargs]):  # 点坐标
            if len(xargs) != self.ndim:
                raise RuntimeError(f"len(xargs)={len(xargs)}!=self.ndim={self.ndim} {xargs}")
            xmin, xmax = self.bbox
            return np.bitwise_and.reduce([((xargs[i] >= xmin[i]) & (xargs[i] <= xmax[i])) for i in range(self.ndim)])
        elif len(xargs) == 1 and isinstance(xargs[0], GeoObject):
            raise NotImplementedError(f"{self.__class__.__name__}.enclose(GeoObject)")
        else:
            return np.bitwise_and.reduce([self.enclose(x) for x in xargs])


_TG = typing.TypeVar("_TG")


class GeoObjectSet(List[_TG], GeoObject):

    def __init__(self,  *args, **kwargs) -> None:
        super().__init__(*args)
        rank = kwargs.pop("rank", None)
        ndim = kwargs.pop("ndim", None)

        if rank is None:
            rank = max([obj.rank for obj in self if isinstance(obj, GeoObject)])

        if ndim is None:
            ndim_list = [obj.ndim for obj in self if isinstance(obj, GeoObject)]
            if len(ndim_list) > 0 and all(ndim_list):
                ndim = ndim_list[0]
            else:
                raise RuntimeError(f"Can not get ndim from {ndim_list}")

        GeoObject.__init__(self, rank=rank, ndim=ndim, **kwargs)

    def __svg__(self) -> str:
        return f"<g id='{self.name}'>\n" + "\t\n".join([g.__svg__() for g in self if isinstance(g, GeoObject)]) + "</g>"

    @property
    def bbox(self) -> BBox: return np.bitwise_or([g.bbox for g in self if isinstance(g, GeoObject)])


def as_geo_object(*args, **kwargs) -> GeoObject:

    if len(kwargs) > 0 or len(args) != 1:
        return GeoObject(*args, **kwargs)
    elif isinstance(args[0], GeoObject):
        return args[0]
    elif isinstance(args[0], collections.abc.Sequence):
        return GeoObjectSet(*args, **kwargs)
    else:
        return GeoObject(*args)
