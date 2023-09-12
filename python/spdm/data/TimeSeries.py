from __future__ import annotations

import collections.abc
import typing

import numpy as np

from ..utils.logger import logger
from ..utils.tags import _not_found_
from ..utils.tree_utils import merge_tree_recursive
from ..utils.typing import array_type, as_array, ArrayType
from .HTree import List
from .sp_property import SpDict, sp_property


class TimeSlice(SpDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    time: float = sp_property(unit="s", type="dynamic", default_value=0.0)  # type: ignore

    def refresh(self, *args, **kwargs) -> TimeSlice:
        super().refresh(*args, **kwargs)
        return self


_T = typing.TypeVar("_T")


class TimeSeriesAoS(List[_T]):
    """
    A series of time slices, each time slice is a state of the system at a given time.
    Each slice is a dict .
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slice_start = 0
        self._slice_stop = 0
        self._slice_current = 0

    @property
    def time(self) -> ArrayType:
        return as_array(
            [getattr(time_slice, "time", None) for time_slice in self._cache]
        )

    def __getitem__(self, idx: int | slice) -> TimeSlice | _T:
        # TODO: 缓存时间片，避免重复创建，减少内存占用

        if isinstance(idx, slice):
            raise NotImplementedError(f"NOT YET! {idx}")

        elif not isinstance(idx, int):
            raise NotImplementedError(f"{idx}")

        elif idx < self._slice_start:
            raise IndexError(f"{idx}<{self._slice_start}")

        elif idx >= self._slice_stop:
            if self._cache is None or self._cache is _not_found_:
                self._cache = []

            self._cache += [None] * (idx - self._slice_stop + 1)

            self._slice_stop = idx

        self._cache[idx - self._slice_start] = self._as_child(
            self._cache[idx - self._slice_start],
            idx,
            entry=self._entry.child(idx) if self._entry is not None else None,
            parent=self._parent,
            default_value=self._default_value,
        )
        return self._cache[idx - self._slice_start]

    def __setitem__(self, idx: int, value):
        raise NotImplementedError(f"")

    @property
    def previous(self) -> TimeSlice | _T:
        return self[self._slice_current - 1]

    @property
    def next(self) -> TimeSlice | _T:
        return self[self._slice_current + 1]

    @property
    def current(self) -> TimeSlice | _T:
        return self[self._slice_current]

    def refresh(self, *args, **kwargs) -> _T:
        self[self._slice_current].refresh(*args, **kwargs)

        self._time[self._slice_current] = self[self._slice_current].time

        return self[self._slice_current]

    def advance(self, *args, **kwargs) -> _T:
        dt = kwargs.pop("dt", None)

        if dt is None:
            pass
        elif len(self) > 0:
            kwargs["time"] = self.current.time + dt
        else:
            kwargs["time"] = dt

        self._slice_current += 1

        return self.refresh(*args, **kwargs)

    def _new_slice(self, *args, time: float, **kwargs):
        self.time.append(time)

        if isinstance(self._default_value, collections.abc.Mapping):
            kwargs.setdefault("default_value", self._default_value)

        if len(args) > 0 and isinstance(args[0], TimeSlice):
            new_obj = args[0]
            new_obj._parent = self._parent
            if time is not None:
                new_obj["time"] = time
        else:
            type_hint = self._type_hint()
            new_obj = type_hint(*args, time=time, **kwargs, parent=self._parent)

        self.insert(new_obj)

        if hasattr(current_slice.__class__, "refresh"):
            current_slice.refresh(*args, **kwargs)
        elif len(args) == 1 and len(kwargs) == 0:
            current_slice.update(args[0])
        else:
            type_hint = self._type_hint(-1)
            new_obj = type_hint(*args, **kwargs, parent=self._parent)
            self[-1] = new_obj
