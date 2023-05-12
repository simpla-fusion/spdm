from __future__ import annotations

import collections.abc
import typing
from functools import cached_property

import numpy as np

from ..utils.misc import builtin_types
from ..utils.Pluggable import Pluggable
from ..utils.typing import ArrayType, NumericType, nTupleType, ArrayLike
from ..utils.logger import logger


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
            _geo_type = kwargs.get("type", Box)

        if isinstance(_geo_type, str):
            _geo_type = [_geo_type,
                         f"spdm.geometry.{_geo_type}#{_geo_type}",
                         f"spdm.geometry.{_geo_type}{cls.__name__}#{_geo_type}{cls.__name__}",
                         f"spdm.geometry.{_geo_type.capitalize()}#{_geo_type.capitalize()}",
                         f"spdm.geometry.{_geo_type.capitalize()}{cls.__name__}#{_geo_type.capitalize()}{cls.__name__}",
                         f"spdm.geometry.{cls.__name__}#{_geo_type}"
                         ]

        super().__dispatch__init__(_geo_type, self, *args, **kwargs)

    def __init__(self, *args, ndims: int = None, rank: int = 0,  **kwargs) -> None:
        if self.__class__ is GeoObject:
            return GeoObject.__dispatch__init__(None, self, *args, **kwargs)
        elif len(args) == 1 and not isinstance(args[0], builtin_types):
            self._impl = args[0]
        elif len(args) > 0:
            raise TypeError(f"illegal {args}")

        self._rank = int(rank)
        self._ndims = ndims if ndims is not None else self._rank
        self._appinfo = kwargs

    def __equal__(self, other: GeoObject) -> bool:
        return isinstance(other, GeoObject) and self._impl == other._impl

    def _repr_svg_(self) -> str:
        return self._impl._repr_svg_() if hasattr(self._impl, "_repr_svg_") else ""

    def __svg__(self) -> str:
        return self._impl._svg() if hasattr(self._impl, "_svg") else ""

    @property
    def ndims(self) -> int: return self._ndims
    """ 几何体所处的空间维度， = 0，1，2，3 ,...  """

    @property
    def rank(self) -> int: return self._rank
    """ 几何体（流形）维度  rank <=ndims

            0: point
            1: curve
            2: surface
            3: volume
            >=4: not defined
    """

    @cached_property
    def bbox(self) -> typing.Tuple[nTupleType, nTupleType]:
        """ bbox of geometry """
        raise NotImplementedError(f"{self.__class__.__name__}.bbox")

    @property
    def center(self) -> np.ndarray: return (np.array(self.bbox[0])+np.array(self.bbox[1]))*0.5
    """ center of geometry """

    @property
    def boundary(self) -> GeoObject[-1]: raise NotImplementedError()
    """ boundary of geometry which is a geometry of rank-1 """

    @property
    def is_convex(self) -> bool: return self._impl.is_convex()
    """ is convex """

    def enclose(self, other) -> bool: return self._impl.encloses(GeoObject(other)._impl)
    """ Return True if all args are inside the geometry, False otherwise. """

    def intersection(self, other) -> typing.Set[GeoObject]:
        """ Return the intersection of self with other. """
        return {GeoObject(o) for o in self._impl.intersection(GeoObject(other)._impl)}

    def reflect(self, line) -> GeoObject:
        """ reflect self by line"""
        return GeoObject(self._impl.reflect(GeoObject(line)._impl))

    def rotate(self, angle, pt=None) -> GeoObject:
        return GeoObject(self._impl.rotate(angle, GeoObject(pt)._impl if pt is not None else None))

    def scale(self, x=1, y=1, pt=None) -> GeoObject:
        """ scale self by x, y, pt """
        return GeoObject(self._impl.scale(x, y, GeoObject(pt)._impl if pt is not None else None))

    def translate(self, *args) -> GeoObject:
        return GeoObject(self._impl.translate(*args))

    def points(self, *uv, **kwargs) -> typing.Sequence[NumericType]:
        """
            coordinates of vertices on mesh
            Return: array-like
                shape = [*shape_of_mesh,ndims]
        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    @property
    def xyz(self) -> np.ndarray:
        """
            coordinates of vertices on mesh
            Return: array-like
                shape = [ndims, *shape_of_mesh]
        """
        return np.moveaxis(self.points(), -1, 0)

    def dl(self, uv=None) -> np.ndarray:
        """
            derivative of shape
            Returns:
                rank==0 : 0
                rank==1 : dl (shape=[n-1])
                rank==2 : dx (shape=[n-1,m-1]), dy (shape=[n-1,m-1])
                rank==3 : dx (shape=[n-1,m-1,l-1]), dy (shape=[n-1,m-1,l-1]), dz (shape=[n-1,m-1,l-1])
        """
        return np.asarray(0)

    def integral(self, func: typing.Callable) -> float:
        return NotImplemented

    # def average(self, func: typing.Callable[[_TCoord, _TCoord], _TCoord]) -> float:
    #     return self.integral(func)/self.length

    @cached_property
    def is_closed(self):
        if self.rank == 0:
            return True
        else:
            return np.allclose(self.xyz[:, 0], self.xyz[:, -1])

    def trim(self):
        return NotImplemented

    def remesh(self, mesh_type=None, /, **kwargs):
        return NotImplemented

    def derivative(self,  *args, **kwargs):
        return NotImplemented

    def pullback(self, func,   *args, **kwargs):
        r"""
            ..math:: f:N\rightarrow M\\\Phi^{*}f:\mathbb{R}\rightarrow M\\\left(\Phi^{*}f\right)\left(u\right)&\equiv f\left(\Phi\left(u\right)\right)=f\left(r\left(u\right),z\left(u\right)\right)
        """

        return NotImplemented

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
    def __init__(self, x_min: ArrayLike = None, x_max: ArrayLike = None, rank=None, ** kwargs) -> None:
        super().__init__(rank=rank if rank is not None else (len(x_min)if x_min is not None else 0), **kwargs)

        self._bbox = (np.asarray(x_min), np.asarray(x_max))

    @property
    def bbox(self) -> typing.Tuple[ArrayType, ArrayType]:
        return self._bbox


_TGSet = typing.TypeVar("_TGSet", bound="GeoObjectSet")


class GeoObjectSet(typing.List[GeoObject | _TGSet]):
    def __init__(self, obj_list=None, *args, **kwargs) -> None:

        if isinstance(obj_list, collections.abc.Sequence) and not isinstance(obj_list, str):
            obj_list = [as_geo_object(obj, *args, **kwargs) for obj in obj_list]
        elif obj_list is None:
            obj_list = []

        super().__init__(obj_list)

    def __svg__(self) -> str:
        raise NotImplementedError(f"{self.__class__.__name__}")

    @property
    def rank(self) -> int:
        return max([obj.rank for obj in self])

    @property
    def ndims(self) -> int:
        return max([obj.ndims for obj in self])

    @property
    def bbox(self) -> typing.Tuple[ArrayType, ArrayType]:
        p_min = np.asarray([min(v.bbox[0][idx] for v in self) for idx in range(self.ndims)])
        p_max = np.asarray([max(v.bbox[1][idx] for v in self) for idx in range(self.ndims)])
        return p_min, p_max


def as_geo_object(*args, **kwargs) -> GeoObject | GeoObjectSet:
    if len(args) == 0:
        return GeoObject(**kwargs)
    elif (isinstance(args[0], GeoObject) or isinstance(args[0], GeoObjectSet)):
        return args[0]
    elif isinstance(args[0], collections.abc.Sequence):
        return GeoObjectSet(*args, **kwargs)
    else:
        return GeoObject(*args, **kwargs)
