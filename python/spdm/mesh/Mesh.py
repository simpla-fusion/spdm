from __future__ import annotations

import collections.abc
import typing
import numpy as np
from functools import cached_property
from enum import Enum
from spdm.utils.typing import ArrayType

from ..geometry.GeoObject import GeoObject, GeoObjectSet, as_geo_object
from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix
from ..utils.Pluggable import Pluggable
from ..utils.typing import ArrayType, NumericType, ScalarType
from ..utils.tags import _not_found_


class Mesh(Pluggable):
    """
    Mesh
    -------
    网格

    @NOTE: In general, a mesh provides more flexibility in representing complex geometries and 
    can adapt to the local features of the solution, while a grid is simpler to generate
    and can be more efficient for certain types of problems.
    """

    _plugin_registry = {}

    @classmethod
    def __dispatch__init__(cls, _mesh_type, self, *args, mesh_type=None, **kwargs) -> None:

        if not _mesh_type:
            _mesh_type = mesh_type

        if isinstance(_mesh_type, Enum) and _mesh_type is not _not_found_:
            _mesh_type = _mesh_type.name

        if isinstance(_mesh_type, str):
            _mesh_type = [_mesh_type,
                          f"spdm.mesh.{_mesh_type}Mesh#{_mesh_type}Mesh",
                          f"spdm.mesh.{_mesh_type.capitalize()}Mesh#{_mesh_type.capitalize()}Mesh"
                          ]
        if _mesh_type is not None and _mesh_type is not _not_found_:
            pass
        elif all([isinstance(arg, (int, np.ndarray)) for arg in args]):
            _mesh_type = "rectilinear"
        else:
            raise RuntimeError(f"Mesh.__dispatch__init__(): mesh_type={_mesh_type} is not found")

        super().__dispatch__init__(_mesh_type, self, *args, **kwargs)

    def __init__(self, *args,  geometry=None, **kwargs) -> None:
        if self.__class__ is Mesh:
            return Mesh.__dispatch__init__(None, self, *args,   geometry=geometry, **kwargs)

        geometry_desc, self._metadata = group_dict_by_prefix(kwargs, "geometry_")

        if isinstance(geometry, collections.abc.Mapping):
            geometry_desc.update(geometry)
            geometry = None
        elif isinstance(geometry, Enum):
            geometry_desc.update({"type": geometry.name})
            geometry = None
        elif isinstance(geometry, str):
            geometry_desc.update({"type": geometry})
            geometry = None
        if isinstance(geometry, (GeoObject, GeoObjectSet)):
            self._geometry = geometry
            if len(geometry_desc) > 0:
                logger.warning(f"self._geometry is specified, ignore geometry_desc={geometry_desc}")
        elif isinstance(geometry_desc, collections.abc.Mapping):
            self._geometry = GeoObject(*args, **geometry_desc)
        else:
            raise RuntimeError(f"Mesh.__init__(): geometry={geometry} is not found, geometry_desc={geometry_desc}")

        self._shape: ArrayType = np.asarray(self._metadata.get("shape", []), dtype=int)

    def __serialize__(self) -> typing.Mapping:
        raise NotImplementedError(f"")

    @classmethod
    def __deserialize__(cls, data: typing.Mapping) -> Mesh:
        raise NotImplementedError(f"")

    @property
    def metadata(self) -> dict: return self._metadata

    @property
    def name(self) -> str: return self.metadata.get("name", 'unamed')

    @property
    def type(self) -> str: return self.metadata.get("type", "unknown")

    @property
    def units(self) -> typing.Tuple[str, ...]: return tuple(self.metadata.get("units", ["-"]))

    @property
    def geometry(self) -> GeoObject | GeoObjectSet: return self._geometry
    """ Geometry of the Mesh  网格的几何形状  """

    @property
    def shape(self) -> typing.Tuple[int, ...]: return self._shape
    """ 存储网格点数组的形状  
        TODO: support multiblock Mesh
        结构化网格 shape   如 [n,m] n,m 为网格的长度dimension
        非结构化网格 shape 如 [<number of vertices>]
    """

    def parametric_coordinates(self, *xyz) -> ArrayType:
        """
            parametric coordinates
            ------------------------
            网格点的 _参数坐标_
            Parametric coordinates, also known as computational coordinates or intrinsic coordinates,
            are a way to represent the position of a point within an element of a mesh.
            一般记作 u,v,w \in [0,1] ,其中 0 表示“起点”或 “原点” origin，1 表示终点end
            mesh的参数坐标(u,v,w)，(...,0)和(...,1)表示边界

            @return: 数组形状为 [geometry.rank, <shape of xyz ...>]的数组
        """
        if len(xyz) == 0:
            return np.stack(np.meshMesh(*[np.linspace(0.0, 1.0, n, endpoint=True) for n in self.shape]))
        else:
            raise NotImplementedError(f"{self.__class__.__name__}.parametric_coordinates for unstructured mesh")

    def coordinates(self, *uvw) -> ArrayType:
        """ 网格点的 _空间坐标_
            @return: _数组_ 形状为 [geometry.dimension,<shape of uvw ...>]
        """
        return self.geometry.coordinates(uvw if len(uvw) > 0 else self.parametric_coordinates())

    def uvw(self, *xyz) -> ArrayType: return self.parametric_coordinates(*xyz)
    """ alias of parametric_coordiantes"""

    def xyz(self, *uvw) -> ArrayType: return self.coordinates(*uvw)
    """ alias of vertices """

    @property
    def vertices(self) -> ArrayType: return self.geometry.coordinates(self.parametric_coordinates())

    @property
    def cells(self) -> typing.Any: raise NotImplementedError(f"{self.__class__.__name__}.cells")
    """ refer to the individual units that make up the mesh"""

    def interpolator(self, y: NumericType, *args, **kwargs) -> typing.Callable[..., NumericType]:
        raise NotImplementedError(f"{self.__class__.__name__}.interpolator")

    def partial_derivative(self, y: NumericType, *args, **kwargs) -> typing.Callable[..., NumericType]:
        raise NotImplementedError(f"{self.__class__.__name__}.partial_derivative")

    def antiderivative(self, y:  NumericType, *args, **kwargs) -> typing.Callable[..., NumericType]:
        raise NotImplementedError(f"{self.__class__.__name__}.antiderivative")

    def integrate(self, y:  NumericType, *args, **kwargs) -> ScalarType:
        raise NotImplementedError(f"{self.__class__.__name__}.integrate")


@Mesh.register(["null", None])
class NullMesh(Mesh):
    def __init__(self, *args, **kwargs) -> None:
        if len(args) > 0 or len(kwargs) > 0:
            raise RuntimeError(f"Ignore args {args} and kwargs {kwargs}")
        super().__init__()


@Mesh.register("regular")
class RegularMesh(Mesh):
    pass


def as_mesh(*args, **kwargs) -> Mesh:
    if len(args) == 1 and isinstance(args[0], Mesh):
        if len(kwargs) > 0:
            logger.warning(f"Ignore kwargs {kwargs}")
        return args[0]
    else:
        return Mesh(*args, **kwargs)
