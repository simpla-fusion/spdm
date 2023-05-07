from __future__ import annotations
from ..utils.Pluggable import Pluggable
from ..utils.misc import builtin_types

import collections.abc
import typing
from functools import cached_property

import numpy as np
from numpy.typing import NDArray, ArrayLike


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
            _geo_type = kwargs.get("geometry_type", None)

            if _geo_type is None and len(args) > 0:
                _geo_type = args[0]
                args = args[1:]

            if isinstance(_geo_type, str):
                _geo_type = [_geo_type,
                             f"spdm.geometry.{_geo_type}#{_geo_type}",
                             f"spdm.geometry.{_geo_type}{cls.__name__}#{_geo_type}{cls.__name__}",
                             f"spdm.geometry.{_geo_type.capitalize()}#{_geo_type.capitalize()}",
                             f"spdm.geometry.{_geo_type.capitalize()}{cls.__name__}#{_geo_type.capitalize()}{cls.__name__}",
                             f"spdm.geometry.{cls.__name__}#{_geo_type}"
                             ]
            else:
                _geo_type = [_geo_type]

            kwargs["geometry_type"] = _geo_type

        super().__dispatch__init__(_geo_type, self, *args, **kwargs)

    def __init__(self, *args,  **kwargs) -> None:
        if self.__class__ is GeoObject:
            return GeoObject.__dispatch__init__(None, self, *args, **kwargs)

        if self.__class__ is GeoObject:
            return GeoObject.__dispatch__init__(None, self, *args, **kwargs)
        elif len(args) == 1 and not isinstance(args[0], builtin_types):
            self._impl = args[0]
        elif len(args) > 0:
            raise TypeError(f"illegal {args}")

        self._appinfo = kwargs

    def __equal__(self, other: GeoObject) -> bool:
        return isinstance(other, GeoObject) and self._impl == other._impl

    def _repr_svg_(self) -> str:
        return self._impl._repr_svg_() if hasattr(self._impl, "_repr_svg_") else ""

    def __svg__(self) -> str:
        return self._impl._svg() if hasattr(self._impl, "_svg") else ""

    def __getitem__(self, *args) -> typing.Any:
        return self._impl.__getitem__(*args)

    @property
    def ndims(self) -> int:
        return self._impl.ambient_dimension

    @property
    def rank(self) -> int:
        """
            0: point
            1: curve
            2: surface
            3: volume
            >=4: not defined
        """
        return self._appinfo.get("rank", None)

    @property
    def bounds(self) -> typing.Tuple[float]:
        return self._impl.bounds

    @property
    def is_convex(self) -> bool:
        return self._impl.is_convex()

    @property
    def center(self) -> np.ndarray:
        return (np.array(self.bounds[::2])+np.array(self.bounds[1::2]))*0.5

    @property
    def boundary(self) -> GeoObject[_I-1]:
        raise NotImplementedError()

    def enclose(self, other) -> bool:
        """ Return True if all args are inside the geometry, False otherwise.
        """
        return self._impl.encloses(GeoObject(other)._impl)

    def intersection(self, other) -> typing.Set[GeoObject]:
        """ Return the intersection of self with other. """
        return {GeoObject(o) for o in self._impl.intersection(GeoObject(other)._impl)}

    def reflect(self, line) -> GeoObject:
        """ reflect self by line"""
        return GeoObject(self._impl.reflect(GeoObject(line)._impl))

    def rotate(self, angle, pt=None) -> GeoObject:
        return GeoObject(self._impl.rotate(angle, GeoObject(pt)._impl if pt is not None else None))

    def scale(self, x=1, y=1, pt=None) -> GeoObject:
        """ scale self by x, y, pt
        """
        return GeoObject(self._impl.scale(x, y, GeoObject(pt)._impl if pt is not None else None))

    def translate(self, *args) -> GeoObject:
        return GeoObject(self._impl.translate(*args))

    def __call__(self, *args: float | np.ndarray, **kwargs) -> np.ndarray | typing.List[float]:
        res = self.points(*args, **kwargs)
        if not isinstance(res, np.ndarray):
            res = res[:]
        return res

    def points(self, *uv, **kwargs) -> typing.Tuple[NDArray]:
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
        # if len(args) == 0:
        #     args = self._mesh

        # return Function(args, func(*self.xyz(*args,   **kwargs)), is_period=self.is_closed)
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


class GeoObject0D(GeoObject):
    @property
    def rank(self) -> int: return 0


class GeoObject1D(GeoObject):
    @property
    def rank(self) -> int: return 1


class GeoObject2D(GeoObject):
    @property
    def rank(self) -> int: return 2


class GeoObject3D(GeoObject):
    @property
    def rank(self) -> int: return 3


_T = typing.TypeVar("_T")


class GeoObjectSet(typing.Set[_T], GeoObject):

    def __init__(self, *args: _T, **kwargs) -> None:
        super().__init__(args)
        GeoObject.__init__(self,  **kwargs)

    @property
    def rank(self) -> int:
        if len(self) == 0:
            raise RuntimeError(f"This is an empyt set!")
        r = np.asarray([v.rank if isinstance(v, GeoObject) else 0 for v in self])
        if not np.all(r == r[0]):
            raise RuntimeError(f"This a mixed GeoObject set!")
        return r[0]
