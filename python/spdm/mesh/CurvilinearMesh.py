import collections.abc
import typing
from functools import cached_property

import numpy as np
from scipy import interpolate

from ..geometry.BSplineSurface import BSplineSurface
from ..geometry.CubicSplineCurve import CubicSplineCurve
from ..geometry.GeoObject import GeoObject, GeoObjectSet
from ..geometry.Point import Point
from ..utils.logger import logger
from ..utils.typing import ArrayType, ScalarType
from .Mesh import Mesh
from .RectilinearMesh import RectilinearMesh


@Mesh.register("curvilinear")
class CurvilinearMesh(RectilinearMesh):
    """
        A `curvilinear Mesh` or `structured Mesh` is a Mesh with the same combinatorial structure as a regular Mesh,
        in which the cells are quadrilaterals or [general] cuboids, rather than rectangles or rectangular cuboids.
            -- [https://en.wikipedia.org/wiki/Regular_Mesh]
    """
    TOLERANCE = 1.0e-5

    def __init__(self, *args, ** kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert isinstance(self._geometry, GeoObjectSet)
        assert isinstance(self._dims, collections.abc.Sequence)

        # if isinstance(geo_mesh, np.ndarray):
        #     if geo_mesh.shape[:-1] != self._shape:
        #         raise ValueError(f"Illegal shape!  {geo_mesh.shape[:-1]} != {self._shape}")
        #     ndims = geo_mesh.shape[-1]
        #     xy = geo_mesh
        #     surf = None
        #     raise NotImplementedError(f"NOT COMPLETE! xy -> surface")
        # elif isinstance(geo_mesh, collections.abc.Sequence) and isinstance(geo_mesh[0], GeoObject):
        #     ndims = geo_mesh[0].ndims
        #     if len(dims[0]) != len(geo_mesh):
        #         raise ValueError(f"Illegal number of sub-surface {len(self)} != {len(geo_mesh)}")
        #     surf = geo_mesh
        # elif isinstance(geo_mesh, GeoObject):
        #     raise NotImplementedError(type(geo_mesh))
        # else:
        #     raise TypeError(
        #         f"geo_mesh should be np.ndarray, typing.Sequence[GeoObject] or GeoObject, not {type(geo_mesh)}")

    def axis(self, idx, axis=0):
        if axis == 0:
            return self._geometry[idx]
        else:
            s = [slice(None, None, None)]*self.ndims
            s[axis] = idx
            s = s+[slice(None, None, None)]

            sub_xy = self.xy[tuple(s)]  # [p[tuple(s)] for p in self._xy]
            sub_uv = [self._uv[(axis+i) % self.geometry.ndim]
                      for i in range(1, self.geometry.ndim)]
            sub_cycle = [self.cycle[(axis+i) % self.geometry.ndim]
                         for i in range(1, self.geometry.ndim)]

            return CurvilinearMesh(sub_xy, sub_uv,  cycle=sub_cycle)

    @property
    def uv(self) -> ArrayType:
        return self._uv

    @cached_property
    def points(self) -> ArrayType:
        if isinstance(self.geometry, GeoObject):
            return self.geometry.points()
        elif isinstance(self.geometry, GeoObjectSet):
            d = np.stack([surf.points(self._dims[1]) for surf in self.geometry], axis=0)
            return tuple(d[..., i] for i in range(self.geometry.ndims))
        else:
            raise RuntimeError(f'Unknown type {type(self.geometry)}')

    @cached_property
    def volume_element(self) -> ArrayType:
        raise NotImplementedError()

    # def pushforward(self, new_uv):
    #     new_shape = [len(u) for u in new_uv]
    #     if new_shape != self.shape:
    #         raise ValueError(f"illegal shape! {new_shape}!={self.shape}")
    #     return CurvilinearMesh(self._xy, new_uv, cycle=self.cycle)

    def interpolator(self, value,  **kwargs):
        if value.shape != self.shape:
            raise ValueError(f"{value.shape} {self.shape}")

        if self.ndims == 1:
            interp = interpolate.InterpolatedUnivariateSpline(self._dims[0], value,  **kwargs)
        elif self.ndims == 2:
            interp = interpolate.RectBivariateSpline(self._dims[0], self._dims[1], value, ** kwargs)
        else:
            raise NotImplementedError(f"NDIMS {self.ndims}>2")
        return interp

    @ cached_property
    def boundary(self):
        return {"inner": self.axis(0, 0),  "outer": self.axis(-1, 0)}

    @ cached_property
    def geo_object(self):
        if self.rank == 1:
            if all([np.var(x)/np.mean(x**2) < CurvilinearMesh.TOLERANCE for x in self.xy.T]):
                gobj = Point(*[x[0] for x in self.xy.T])
            else:
                gobj = CubicSplineCurve(self.xy, self._uv[0], is_closed=self.cycle[0])
        elif self.rank == 2:
            gobj = BSplineSurface(self.xy, self._uv,  is_closed=self.cycle)
        else:
            raise NotImplementedError()

        return gobj
