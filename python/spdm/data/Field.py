from __future__ import annotations

import collections.abc
import functools
import typing
from enum import Enum
import numpy as np
from ..mesh.Mesh import Mesh
from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix
from ..utils.tags import _not_found_
from ..utils.typing import ArrayType, array_type, as_array
from ..utils.tree_utils import merge_tree_recursive
from .Expression import Expression
from .Functor import Functor
from .Path import Path


def guess_mesh(holder, prefix="mesh", **kwargs):
    if holder is None or holder is _not_found_:
        return None

    metadata = getattr(holder, "_metadata", {})

    mesh, *_ = group_dict_by_prefix(metadata, prefix, sep=None)

    if mesh is None:
        coordinates, *_ = group_dict_by_prefix(metadata, "coordinate", sep=None)

        if coordinates is not None:
            coordinates = {int(k): v for k, v in coordinates.items() if k.isdigit()}
            coordinates = dict(sorted(coordinates.items(), key=lambda x: x[0]))
            coordinates = [Path(c).fetch(holder) for c in coordinates.values()]
            mesh = {"dims": coordinates}

    elif isinstance(mesh, str):
        mesh = holder.get(mesh, _not_found_)

    # if all([isinstance(c, str) and c.startswith("../grid") for c in coordinates.values()]):
    #     o_mesh = getattr(holder, "grid", None)
    #     if isinstance(o_mesh, Mesh):
    #         # if self._mesh is not None and len(self._mesh) > 0:
    #         #     logger.warning(f"Ignore {self._mesh}")
    #         self._domain = o_mesh
    #     elif isinstance(o_mesh, collections.abc.Sequence):
    #         self._domain = merge_tree_recursive(self._domain, {"dims": o_mesh})
    #     elif isinstance(o_mesh, collections.abc.Mapping):
    #         self._domain = merge_tree_recursive(self._domain, o_mesh)
    #     elif o_mesh is not None:
    #         raise RuntimeError(f"holder.grid is not a Mesh, but {type(o_mesh)}")
    # else:
    #     dims = tuple([(holder.get(c) if isinstance(c, str) else c) for c in coordinates.values()])
    #     self._domain = merge_tree_recursive(self._domain, {"dims": dims})
    elif isinstance(mesh, Enum):
        mesh = {"type": mesh.name}

    elif isinstance(mesh, collections.abc.Sequence) and all(isinstance(d, array_type) for d in mesh):
        mesh = {"dims": mesh}

    elif isinstance(mesh, collections.abc.Mapping):
        pass

    if mesh is None or mesh is _not_found_:
        return guess_mesh(getattr(holder, "_parent", None), prefix=prefix, **kwargs)
    else:
        return mesh


class Field(Expression):
    """Field
    ---------
    Field 是 Function 在流形（manifold/Mesh）上的推广， 用于描述流形上的标量场，矢量场，张量场等。

    Field 所在的流形记为 mesh ，可以是任意维度的，可以是任意形状的，可以是任意拓扑的，可以是任意坐标系的。

    Mesh 网格描述流形的几何结构，比如网格的拓扑结构，网格的几何结构，网格的坐标系等。

    Field 与 Function的区别：
        - Function 的 mesh 是一维数组表示dimensions/axis
        - Field 的 mesh 是 Mesh，可以表示复杂流形上的场等。


    """

    Domain = Mesh

    def __init__(self, *xy, **kwargs):
        if len(xy) == 0:
            raise RuntimeError(f"illegal x,y {xy} ")

        value = xy[-1]

        dims = xy[:-1]

        if isinstance(value, (Functor, Expression)) or callable(value):
            func = value
            value = None
        else:
            func = None
            value = as_array(value)

        mesh, kwargs = group_dict_by_prefix(kwargs, prefixes="mesh")

        if mesh is not None:
            if len(dims) == 0:
                pass
            elif isinstance(mesh, dict):
                mesh["dims"] = dims
            else:
                raise RuntimeError(f"'mesh' is defined, ignore dims={dims} {mesh}")

        elif "domain" in kwargs:
            mesh = kwargs["domain"]
            if len(dims) > 0:
                raise RuntimeError(f"'mesh' is defined, ignore dims={dims}")
        else:
            mesh = dims

        super().__init__(func, domain=mesh, **kwargs)

        self._value = value
        self._ppoly = None

    def _repr_svg_(self) -> str:
        from ..view.View import display

        return display(
            (
                (*self.mesh.points, self.__array__()),
                {
                    "label": self.__label__,
                    "axis_label": self.mesh.axis_label,
                },
            ),
            output="svg",
        )

    @property
    def mesh(self) -> Mesh:
        if isinstance(self._domain, Mesh):
            return self._domain

        if self._domain is None or self._domain is _not_found_:
            self._domain = guess_mesh(self, prefix="mesh")

        if not isinstance(self._domain, Mesh):
            mesh_desc, *_ = group_dict_by_prefix(self._metadata, prefixes="mesh", sep="_")
            self._domain = Mesh(self._domain, parent=self, **(mesh_desc or {}))

        return self._domain

    @property
    def domain(self) -> Mesh:
        return self.mesh

    def ppoly(self):
        if self._ppoly is None:
            self._ppoly = self.mesh.interpolator(self.__array__())
        return self._ppoly

    def __array__(self, *args, **kwargs) -> ArrayType:
        if self._value is None or self._value is _not_found_:
            self._value = super().__array__()
        return self._value

    def __functor__(self) -> typing.Callable[..., ArrayType]:
        if self._func is None and self._value is not None:
            self._func = self.ppoly()
        return self._func

    def grad(self, n=1) -> Field:
        ppoly = self.__functor__()

        if isinstance(ppoly, tuple):
            ppoly, opts = ppoly
        else:
            opts = {}

        if self.mesh.ndim == 2 and n == 1:
            return Field(
                (ppoly.partial_derivative(1, 0), ppoly.partial_derivative(0, 1)),
                mesh=self.mesh,
                name=f"\\nabla({self.__str__()})",
                **opts,
            )
        elif self.mesh.ndim == 3 and n == 1:
            return Field(
                (
                    ppoly.partial_derivative(1, 0, 0),
                    ppoly.partial_derivative(0, 1, 0),
                    ppoly.partial_derivative(0, 0, 1),
                ),
                mesh=self.mesh,
                name=f"\\nabla({self.__str__()})",
                **opts,
            )
        elif self.mesh.ndim == 2 and n == 2:
            return Field(
                (ppoly.partial_derivative(2, 0), ppoly.partial_derivative(0, 2), ppoly.partial_derivative(1, 1)),
                mesh=self.mesh,
                name=f"\\nabla^{n}({self.__str__()})",
                **opts,
            )
        else:
            raise NotImplemented(f"TODO: ndim={self.mesh.ndim} n={n}")

    def derivative(self, d, *args, **kwargs) -> Field:
        if isinstance(d, int) and d < 0:
            func = self.ppoly().antiderivative(*d)
            return Field(func, mesh=self.mesh, name=f"I_{d}({self})")
        elif isinstance(d, collections.abc.Sequence):
            func = self.ppoly().partial_derivative(*d)
            return Field(func, mesh=self.mesh, name=f"d_{d}({self})")
        else:
            func = self.ppoly().derivative(d)
            return Field(func, mesh=self.mesh, name=f"d_{d}({self})")
