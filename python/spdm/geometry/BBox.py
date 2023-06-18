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

    def __svg__(self, **kwargs):
        if self.ndim != 2:
            raise NotImplementedError(f"{self.__class__.__name__}.__svg__ ndim={self.ndim}")
        else:
            xmin = self._xmin
            xmax = self._xmax
            if np.allclose(self._xmin, self._xmax):
                return f"<circle cx='{xmin[0]}' cy='{xmin[1]}' r='3' stroke='black' stroke-width='1' fill='red' />"
            else:
                return f"<rect x='{xmin[0]}' y='{xmin[1]}' width='{xmax[0]-xmin[0]}' height='{xmax[1]-xmin[1]}' stroke='black' stroke-width='1' fill='none' />"

    @property
    def is_valid(self) -> bool: return np.all(self._xmin < self._xmax) == True

    @property
    def is_degraded(self) -> bool: return (self.is_valid and np.any(np.isclose(self._xmin, self._xmax))) == True

    def __equal__(self, other: BBox) -> bool:
        return np.allclose(self._xmin, other._xmin) and np.allclose(self._xmax, other._xmax)

    def __or__(self, other: BBox) -> BBox:
        if other is None:
            return self
        else:
            return BBox(np.min(self._xmin, other._xmin), np.max(self._xmax, other._xmax))

    def __and__(self, other: BBox) -> BBox | None:
        if other is None:
            return None
        else:
            res = BBox(np.max(self._xmin, other._xmin), np.min(self._xmax, other._xmax))
            return res if res.is_valid else None

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
