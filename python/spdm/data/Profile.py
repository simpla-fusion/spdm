from __future__ import annotations

import collections.abc
import pprint
import typing
from enum import Enum
from functools import cached_property

import numpy as np

from ..mesh.Mesh import Mesh
from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix
from ..utils.tags import _not_found_
from ..utils.typing import ArrayType, NumericType
from .Container import Container
from .Expression import Expression
from .Function import Function
from .Node import Node
from .Profile import Profile

_T = typing.TypeVar("_T")


class Profile(Function[_T], Node):
    """
    Profile
    ---------
    Profile= Function + Node 是具有 Function 特性的 Node。
    Function 的 value 由 Node.__value__ 提供，

    mesh 可以根据 kwargs 中的 coordinate* 从 Node 所在 Tree 获得

    """

    def __init__(self,  value: NumericType | Expression, *dims, mesh=None, opts=None, metadata=None, **kwargs) -> None:
        """
            Parameters
            ----------
            value : NumericType
                函数的值
            dims : typing.Any
                用于坐标轴(dims)，用于构建 Mesh
            kwargs : typing.Any
                    coordinate* : 给出各个坐标轴的path, 用于从 Node 所在 Tree 获得坐标轴(dims)
                    op_*        : 用于传递给运算符的参数
                    *           : 用于传递给 Node 的参数

        """
        if metadata is None:
            metadata = {}

        if hasattr(value.__class__, "__entry__"):
            Node.__init__(self, value, metadata=metadata, **kwargs)
            value = None
        else:
            Node.__init__(self, None,  metadata=metadata, **kwargs)

        if len(dims) > 0:
            if "coordinate1" in metadata:
                logger.warning(f"Ignore coordinate*={kwargs['coordinate1']}!")
        else:
            coordinates = {int(k[10:]): v for k, v in metadata.items()
                           if k.startswith("coordinate") and k[10:].isdigit()}
            coordinates = dict(sorted(coordinates.items(), key=lambda x: x[0]))

            if len(coordinates) > 0 and self._parent is None:
                raise RuntimeError(f"Can not determint the coordinates from DataTree!")
            elif all([isinstance(c, str) and c.startswith('../grid') for c in coordinates.values()]):
                if mesh is not None:
                    logger.warning(f"Ignore mesh={mesh}  !")
                mesh = self._parent.grid
                dims = ()
            else:
                dims = tuple([(self._find_node_by_path(c) if isinstance(c, str) and c.startswith('../') else c)
                              for c in coordinates.values()])

        Function.__init__(self, value, *dims, mesh=mesh, **(opts if opts is not None else {}))

    @property
    def metadata(self) -> dict: return self._metadata

    @property
    def name(self) -> str: return self._metadata.get("name", None)

    def __str__(self) -> str: return Function.__str__(self)

    @property
    def data(self) -> ArrayType: return self.__array__()

    @property
    def points(self) -> typing.List[ArrayType]:
        if len(self._mesh) == 1:
            return self._mesh
        else:
            return np.meshgrid(*self._mesh, indexing="ij")

    # def __value__(self) -> ArrayType:
    #     if not isinstance(self._value, np.ndarray) and not self._value:
    #         self._value = Node.__value__(self)
    #         if not isinstance(self._value, np.ndarray) and not self._value:
    #             raise RuntimeError(f"Can not evaluate {self} {self._entry}!")
    #     return self._value

    def __value__(self) -> ArrayType:
        value = Node.__value__(self)
        if isinstance(value, np.ndarray):
            return value
        elif not value:
            if self._op is not None:
                value = super().__call__(*self.points)
            else:
                raise RuntimeError(f"Field.__array__(): value is not found!")

        if not isinstance(value, np.ndarray):
            value = np.asarray(value, dtype=self.__type_hint__)
        self._cache = value
        return value
