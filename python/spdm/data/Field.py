from __future__ import annotations

import collections.abc
import typing
from enum import Enum

import numpy as np

from ..mesh.Mesh import Mesh
from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix
from ..utils.tags import _not_found_
from ..utils.typing import ArrayType, NumericType
from .Function import Function
from .Expression import Expression
from .Node import Node

_T = typing.TypeVar("_T")


class Field(Node, Function[_T]):
    """ Field
        ---------
        Field 是 Function 在流形（manifold）上的推广， 用于描述流形上的标量场，矢量场，张量场等。

        Field 所在的流形记为 domain，可以是任意维度的，可以是任意形状的，可以是任意拓扑的，可以是任意坐标系的。

        Field 的 domain，用Mesh来描述流形的几何结构，比如网格的拓扑结构，网格的几何结构，网格的坐标系等。

        domain/mesh_*： 网格

    """

    def __init__(self, value: typing.Any, *args, domain=None,   **kwargs):
        """
            Parameters
            ----------
            value : NumericType
                函数的值
            domain : typing.List[ArrayType]
                函数的定义域
            args : typing.Any
                位置参数, 用于与mesh_*，coordinate* 一起构建 mesh
            kwargs : typing.Any
                命名参数，
                    mesh_*      : 用于传递给网格的参数
                    coordinate* : 给出各个坐标轴的path
                    op_*        : 用于传递给运算符的参数
                    *           : 用于传递给 Node 的参数

        """

        domain_desc,  kwargs = group_dict_by_prefix(kwargs,  "mesh_")

        if callable(value) or (isinstance(value, tuple) and callable(value[0])):
            Node.__init__(self, None,   **kwargs)
        else:
            Node.__init__(self, value,   **kwargs)
            value = None

        if isinstance(domain, (Mesh, np.ndarray, tuple)):
            if len(domain_desc) > 0 or len(args) > 0:
                logger.warning(f"Ignore domain={domain_desc} and args={args}!")
        elif domain is not None and domain is not _not_found_:
            raise TypeError(f"Illegal mesh type {type(domain)} !")
        else:
            if isinstance(domain, collections.abc.Mapping):
                domain_desc.update(domain)
            elif isinstance(domain, Enum):
                domain_desc.update({"type": domain.name})
            elif isinstance(domain, str):
                domain_desc.update({"type": domain})

            coordinates = {k[10:]: v for k, v in self._metadata.items() if k.startswith("coordinate")}
            coordinates = {int(k): v for k, v in coordinates.items() if k.isdigit()}
            coordinates = dict(sorted(coordinates.items(), key=lambda x: x[0]))

            if len(coordinates) == 0:
                domain = Mesh(*args, **domain_desc)
            elif len(args) > 0:
                raise RuntimeError(f"Coordiantes is defined twice!  len(args)={len(args)}")
            elif all(p.startswith("../grid/dim") for p in coordinates.values()):
                domain = self._parent.grid
            else:
                coordinates = tuple([(slice(None) if (c == "1...N")
                                      or not isinstance(c, str) else self._find_node_by_path(c)) for c in coordinates.values()])

                domain = Mesh(*coordinates, **domain_desc)

        Function.__init__(self, value, domain=domain)

    def __str__(self) -> str: return Function.__str__(self)

    @property
    def mesh(self) -> Mesh: return self._domain
    """ alias of domain
        网格，用来描述流形的几何结构，比如网格的拓扑结构，网格的几何结构，网格的坐标系等。 """

    @property
    def bbox(self): return self.mesh.geometry.bbox

    @property
    def name(self) -> str: return self._metadata.get("name", 'unnamed')

    def __value__(self) -> ArrayType:
        value = Node.__value__(self)
        if isinstance(value, np.ndarray):
            return value

        if not isinstance(value, np.ndarray) and not value:
            if self._op is not None:
                value = super().__call__(*self.mesh.xyz)
            else:
                raise RuntimeError(f"Field.__array__(): value is not found!")

        if not isinstance(value, np.ndarray):
            value = np.asarray(value, dtype=self.__type_hint__)
        self._cache = value
        return value

    def __array__(self) -> ArrayType:
        res = self.__value__()
        assert (isinstance(res, np.ndarray))
        return res

    def _eval(self, *args, **kwargs) -> NumericType:
        if self._op is None:
            self._op = self.mesh.interpolator(self.__array__())

        if len(args) == 0:
            args = self.mesh.xyz

        if all([isinstance(a, np.ndarray) for a in args]):
            shape = args[0].shape
            return super()._eval(*[a.ravel() for a in args],  **kwargs).reshape(shape)
        else:
            return super()._eval(*args,  **kwargs)

    def compile(self) -> Field[_T]:
        return Field[_T](self.__array__(), domain=self.domain, metadata=self._metadata)

    def partial_derivative(self, *d) -> Field[_T]:
        if hasattr(self._op, "partial_derivative"):
            res = Field[_T](self._op.partial_derivative(*d),  domain=self.mesh,
                            metadata={"name": f"d{self.name}_d{d}"})
        else:
            res = Field[_T](self.mesh.partial_derivative(self.__array__(), *d), domain=self.mesh,
                            metadata={"name": f"d{self.name}_d{d}"})
        return res

    def pd(self, *d) -> Field: return self.partial_derivative(*d)

    def antiderivative(self, *d) -> Field:
        if hasattr(self._op, "antiderivative"):
            res = Field[_T](self._op.antiderivative(*d), domain=self.mesh,
                            metadata={"name": f"\int {self.name} d{d}"})
        else:
            res = Field[_T](self.mesh.antiderivative(self.__array__(), *d), domain=self.mesh,
                            metadata={"name": f"\int {self.name} d{d}"})
        return res
