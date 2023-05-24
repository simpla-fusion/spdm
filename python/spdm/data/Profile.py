from __future__ import annotations

import collections.abc
import pprint
import typing
from enum import Enum
from functools import cached_property

import numpy as np

from spdm.utils.typing import ArrayType

from ..mesh.Mesh import Mesh
from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix
from ..utils.tags import _not_found_
from ..utils.typing import ArrayType, NumericType, numeric_type
from .Container import Container
from .Expression import Expression
from .Function import Function
from .Node import Node

_T = typing.TypeVar("_T")


class Profile(Function[_T], Node):
    """
    Profile
    ---------
    Profile= Function + Node 是具有 Function 特性的 Node。
    Function 的 value 由 Node.__value__ 提供，

    mesh 可以根据 kwargs 中的 coordinate* 从 Node 所在 Tree 获得

    """

    def __init__(self,  value: NumericType | Expression, *dims, mesh=None, metadata=None, parent=None,  **kwargs) -> None:
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
        if hasattr(value.__class__, "__entry__"):
            entry = value
            value = None
        else:
            entry = None

        if metadata is None:
            metadata = {}

        opts, kwargs = group_dict_by_prefix(kwargs, "op_")

        Function.__init__(self, value, *dims, mesh=mesh, **opts)

        Node.__init__(self, entry,  metadata=metadata, parent=parent, **kwargs)

    @property
    def metadata(self) -> dict: return self._metadata

    @property
    def name(self) -> str: return self._metadata.get("name", "unnamed")

    @property
    def coordinates(self) -> typing.List[ArrayType]: return self.points

    @property
    def data(self) -> ArrayType: return self.__array__()

    def _refresh(self) -> None:
        if not self.is_empty:
            return

        parent = self._parent
        metadata = self._metadata
        if len(dims) > 0:
            if "coordinate1" in metadata:
                logger.warning(f"Ignore coordinate*={metadata['coordinate1']} dims={dims}!")
            if not mesh:
                mesh = dims
            else:
                logger.warning(f"Ignore mesh={type(mesh)} dims={dims}!")
        elif isinstance(parent, Node):

            coordinates = {int(k[10:]): v for k, v in metadata.items()
                           if k.startswith("coordinate") and k[10:].isdigit()}

            coordinates = dict(sorted(coordinates.items(), key=lambda x: x[0]))

            if len(coordinates) > 0 and (parent is None or parent is _not_found_):
                if not mesh:
                    raise RuntimeError(f"Can not determint the coordinates from DataTree! {coordinates} {type(parent)}")
            elif len(coordinates) > 0 and all([isinstance(c, str) and c.startswith('../grid') for c in coordinates.values()]):
                if mesh is not None and mesh is not parent.grid:
                    logger.warning(f"Ignore mesh={mesh}  !")
                mesh = getattr(parent, "grid", None)
                dims = ()
            else:
                dims = tuple([(parent._find_node_by_path(c, prefix="../") if isinstance(c, str) else c)
                              for c in coordinates.values()])
        v = Node.__value__(self)
        if callable(v):
            op = v
            value = None
        else:
            value = v
            op = None
            Function.__init__(self, v, *self.dims, mesh=self.mesh)
        if self._value is None and self._op is None and len(self._expr_nodes) == 0:
            pass

    def __value__(self) -> ArrayType:
        self._refresh()
        return super()._value

    def __call__(self, *args, **kwargs) -> _T | ArrayType:
        self._refresh()
        return super().__call__(*args, **kwargs)
