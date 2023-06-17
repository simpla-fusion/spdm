from __future__ import annotations

import collections.abc
import typing
import uuid
from copy import copy
from functools import cached_property

import numpy as np

from ..data.Expression import Expression
from ..data.List import List
from ..utils.logger import logger
from ..utils.Pluggable import Pluggable
from ..utils.typing import (ArrayLike, ArrayType, NumericType, ScalarType,
                            array_type, nTupleType, numeric_type)


class BBox:
    def __init__(self, xmin: ArrayLike, xmax: ArrayLike) -> None:
        self._xmin = np.asarray(xmin)
        self._xmax = np.asarray(xmax)

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
        if len(args) == 1 and hasattr(args[0], "bbox"):
            return self.enclose(args[0].bbox)
        elif len(args) == 1 and isinstance(args[0], BBox):
            other: BBox = args[0]
            return np.all(self._xmin <= other._xmin) & np.all(self._xmax >= other._xmax)
        else:
            return np.bitwise_and.reduce([((self._xmin[idx] <= x) & (x <= self._xmax[idx])) for idx, x in enumerate(args)])

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
