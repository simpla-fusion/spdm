from __future__ import annotations

import collections.abc
import typing

import numpy as np

from ..utils.logger import logger
from ..utils.tags import _not_found_
from ..utils.typing import ArrayType, array_type, as_array
from .Entry import Entry
from .HTree import List, HTree
from .sp_property import SpTree, sp_property
from .Path import update_tree


class TimeSlice(SpTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    time: float = sp_property(unit="s", default_value=0.0)  # type: ignore

    def refresh(self, *args, **kwargs):
        self.update(*args, **kwargs)


_TSlice = typing.TypeVar("_TSlice", bound=TimeSlice)


class TimeSeriesAoS(List[_TSlice]):
    """
    A series of time slices .

    用以管理随时间变化（time series）的一组状态（TimeSlice）。

    current:
        指向当前时间片，即为序列最后一个时间片吗。

    TODO:
      1. 缓存时间片，避免重复创建，减少内存占用
      2. 缓存应循环使用
      3. cache 数据自动写入 entry 落盘
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._entry_cursor = None
        self._cache_cursor = 0
        self._cache_depth = kwargs.pop("cache_depth", 3)

        if self._cache is _not_found_ or self._cache is None or len(self._cache) == 0:
            self._cache = [_not_found_] * self._cache_depth

        else:
            if len(self._cache) < self._cache_depth:
                self._cache += [_not_found_] * (self._cache_depth - len(self._cache))
            else:
                self._cache_depth = len(self._cache)

            self._cache_cursor = len(self._cache) - 1

    def dump(self, entry: Entry, **kwargs) -> None:
        """将数据写入 entry"""
        entry.insert([{}] * len(self._cache))
        for idx, value in enumerate(self._cache):
            if isinstance(value, HTree):
                value.dump(entry.child(idx), **kwargs)
            else:
                entry.child(idx).insert(value)

    @property
    def time(self) -> float:
        return self.current.time

    @property
    def dt(self) -> float:
        return self._metadata.get("dt", 0.1)

    @property
    def current(self) -> _TSlice:
        return self._get(0)

    @property
    def previous(self) -> _TSlice:
        return self._get(-1)

    @property
    def is_initializied(self) -> bool:
        return self._entry_cursor is not None

    def _find_slice_by_time(self, time) -> typing.Tuple[int, float]:
        if self._entry is None:
            return None, None

        time_coord = getattr(self, "_time_coord", _not_found_)
        if time_coord is _not_found_:
            time_coord = self._metadata.get("coordinate1", _not_found_)

        if isinstance(time_coord, str):
            time_coord = self.get(time_coord, default_value=_not_found_)
            if time_coord is None:
                time_coord = self._entry.child(time_coord).fetch()

        self._time_coord = time_coord

        pos = None

        if isinstance(time_coord, np.ndarray):
            indices = np.where(time_coord < time)[0]
            if len(indices) > 0:
                pos = indices[-1] + 1
                time = time_coord[pos]

        elif self._entry is not None:
            pos = self._entry_cursor or 0

            while True:
                t_time = self._entry.child(f"{pos}/time").fetch(default_value=_not_found_)

                if t_time is _not_found_ or t_time is None or t_time > time:
                    time = None
                    break
                elif np.isclose(t_time, time):
                    time = t_time
                    break
                else:
                    pos = pos + 1

        return pos, time

    def _get(self, idx: int, *args, **kwargs) -> _TSlice:
        if not isinstance(idx, int):
            return _not_found_
        elif not self.is_initializied:
            self.initialize(*args, **kwargs)

        cache_pos = (self._cache_cursor + idx + self._cache_depth) % self._cache_depth

        value = self._cache[cache_pos]

        if not (value is _not_found_ or isinstance(value, TimeSlice)):
            entry = self._entry.child(self._entry_cursor + idx) if isinstance(self._entry, Entry) else None
            value = self._as_child(value, self._entry_cursor + idx, _entry=entry, _parent=self._parent)
            self._cache[cache_pos] = value

        return value

    def initialize(self, *args, **kwargs):
        if self.is_initializied:
            return
            # raise RuntimeError(f"TimeSeries is already initialized!")

        self._cache_cursor = 0

        self._cache[self._cache_cursor] = update_tree(self._cache[self._cache_cursor], *args, kwargs)

        current = self._cache[self._cache_cursor]

        if isinstance(current, dict):
            time = current.get("time", None)
        else:
            time = getattr(current, "time", None)

        if time is None:
            time = 0.0

        self._entry_cursor, time_hint = self._find_slice_by_time(time)

        if time_hint is not None:
            self._cache[self._cache_cursor] = update_tree(current, {"time": time_hint})

    def refresh(self, *args, **kwargs) -> typing.Type[TimeSeriesAoS]:
        if not self.is_initializied:
            self.initialize(*args, **kwargs)
        else:
            self._cache[self._cache_cursor] = update_tree(self._cache[self._cache_cursor], *args, **kwargs)

    def advance(self, *args, **kwargs) -> _TSlice:
        if not self.is_initializied:
            self.initialize(*args, **kwargs)
        else:
            self._cache_cursor = (self._cache_cursor + 1) % self._cache_depth

            self._entry_cursor += 1

            self._cache[self._cache_cursor] = _not_found_

            self.refresh(*args, **kwargs)

        return self.current
