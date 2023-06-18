from __future__ import annotations

import abc
import collections.abc
import typing
import functools
import numpy as np

from ..utils.logger import logger
from ..utils.typing import ArrayLike, ArrayType, NumericType, nTupleType
from .GeoObject import GeoObject
from .BBox import BBox
from .Point import Point
from .Line import Segment


class PointSet(GeoObject):
    def __init__(self, *args,  **kwargs) -> None:
        if len(args) == 1:
            if isinstance(args[0], collections.abc.Sequence):
                points = np.vstack(args)
            elif isinstance(args[0], np.ndarray):
                points = args[0]
            else:
                raise TypeError(f"illegal type args[0]={type(args[0])}")
        else:
            points = np.vstack(args)

        coordinates = kwargs.pop("coordinates", None)

        super().__init__(ndim=points.shape[-1], ** kwargs)

        self._points: np.ndarray = points
        self._edges = None

        if coordinates is not None:
            if isinstance(coordinates, str):
                coordinates = [x.strip() for x in coordinates.split(",")]

            if len(coordinates) != self._ndim:
                raise ValueError(f"coordinates {coordinates} not match ndim {self._ndim}")
            elif isinstance(coordinates, collections.abc.Sequence):
                for idx, coord_name in enumerate(coordinates):
                    setattr(self, coord_name, self._points[..., idx])

    def __copy__(self) -> PointSet:
        other: PointSet = super().__copy__()  # type:ignore
        other._points = self._points
        return other

    @property
    def points(self) -> typing.List[ArrayType]: return [self._points[..., idx] for idx in range(self.ndim)]

    @functools.cached_property
    def bbox(self) -> BBox:
        return BBox([np.min(self._points[..., idx]) for idx in range(self.ndim)],
                    [np.max(self._points[..., idx]) for idx in range(self.ndim)])

    @property
    def boundary(self) -> PointSet | None:
        if self.is_closed:
            return None
        elif self._points.ndim == 2:
            if self.rank == 2:
                return PointSet(self._points)
            elif self.rank == 1:
                return PointSet((self._points[0], self._points[1]))
        else:
            raise NotImplementedError(f"{self.__class__.__name__}.boundary")

    @property
    def vertices(self) -> typing.Generator[Point, None, None]:
        if self._points.ndim != 2:
            raise NotImplementedError(f"{self.__class__.__name__}.vertices for ndim!=2")

        for p in self._points:
            yield Point(*p)

    @property
    def edges(self) -> typing.Generator[Segment, None, None]:
        if self._points.ndim != 2:
            raise NotImplementedError(f"{self.__class__.__name__}.edges for ndim!=2")
        elif self._edges is None:
            for idx in range(self._points.shape[0]-1):
                yield Segment(self._points[idx], self._points[idx+1])
        elif isinstance(self._edges, np.ndarray):
            for b, e in self._edges:
                yield Segment(self._points[b], self._points[e])
        else:
            raise NotImplementedError()
