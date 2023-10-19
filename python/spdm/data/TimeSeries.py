from __future__ import annotations

import collections.abc
import typing

import numpy as np

from ..utils.logger import logger
from ..utils.tags import _not_found_
from ..utils.tree_utils import update_tree
from ..utils.typing import ArrayType, array_type, as_array
from .Entry import Entry
from .HTree import List, HTree
from .sp_property import SpTree, sp_property


class TimeSlice(SpTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    time: float = sp_property(unit="s",   default_value=0.0)  # type: ignore


_TSlice = typing.TypeVar("_TSlice")  # , TimeSlice, typing.Type[TimeSlice]


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

        self._cache_cursor = 0
        self._cache_depth = self._metadata.get("cache_depth", 3)

        if self._cache is _not_found_ or self._cache is None:
            self._cache = [None]*self._cache_depth

        elif len(self._cache) < self._cache_depth:
            self._cache += [None]*(self._cache_depth-len(self._cache))

        self._time_coord = None

    def dump(self, entry: Entry, **kwargs) -> None:
        """ 将数据写入 entry """
        entry.insert([{}]*len(self._cache))
        for idx, value in enumerate(self._cache):
            if isinstance(value, HTree):
                value.dump(entry.child(idx), **kwargs)
            else:
                entry.child(idx).insert(value)

    @property
    def time(self) -> float: return self.current.time

    @property
    def dt(self) -> float: return self._metadata.get("dt", 0.1)

    @property
    def current(self) -> typing.Typle[TimeSlice]: return self._get(0)

    @property
    def prev(self) -> _TSlice: return self._get(-1)

    def _find_slice_by_time(self, time) -> Entry:

        time_coord = self._time_coord or self._metadata.get("coordinate1", "../time")

        if isinstance(time_coord, str):
            self._time_coord = self.get(time_coord, default_value=None) or self._entry.child(time_coord).fetch()

        entry = None

        if isinstance(self._time_coord, np.ndarray):
            indices = np.where(self._time_coord <= time)[0]
            if len(indices) > 0:
                pos = indices[-1]
                time = self._time_coord[pos]
            entry = self._entry.child(pos)

        elif self._entry is not None:
            current = self._cache[self._cache_cursor]
            if current is None and self._entry is not None:
                entry = self._entry.child(0)
            elif isinstance(current, TimeSlice) and current.time < time:
                entry = self.current._entry

            while entry is not None:
                t_time = entry.child("time").fetch(default_value=_not_found_)
                if t_time is _not_found_ or t_time is None:
                    break
                elif t_time >= time:
                    time = t_time
                    break

                entry = entry.next()

        return entry, time

    def _get(self, idx: int, **kwargs) -> _TSlice:

        if not isinstance(idx, int):
            return _not_found_

        idx = (self._cache_cursor + idx + self._cache_depth) % self._cache_depth

        value = self._cache[idx]

        entry = None

        if isinstance(value, TimeSlice) or (self._entry is None or self._entry is _not_found_):
            pass

        elif isinstance(value, Entry):
            entry = value
            value = None

        elif isinstance(value, dict) and "time" in value:
            entry, time = self._find_slice_by_time(value.get("time"))
            value["time"] = time

        elif idx == self._cache_cursor:
            if self._entry is not None:
                entry = self._entry.child(idx)

        elif self.current._entry is not None:
            entry = self.current._entry.next(idx-self._cache_cursor)

        elif self._entry is not None:
            entry = self._entry.child(idx)

        obj = self._as_child(value, None, _entry=entry, _parent=self._parent)

        self._cache[idx] = obj

        return obj  # type:ignore

    def _flush(self, start: int = 0, end: int = None) -> None:
        start = start or self._cache_cursor
        end = end or start+1

        if isinstance(self._entry, Entry) and self._entry.is_writable:
            start_slice = self._current_slice - self._cache_depth + 1
            for idx in range(start, end):
                self._entry.child(start_slice+idx).update(self._cache[idx].dump())

        for idx in range(start, end):
            self._cache[idx] = None

    def refresh(self, *args, **kwargs) -> _TSlice:

        current = self._cache[self._cache_cursor]

        if isinstance(current, HTree):
            current.update(*args, **kwargs)

        else:
            update_tree(self._cache, self._cache_cursor, *args, **kwargs)

        # return self.current

    def advance(self, *args, **kwargs) -> _TSlice:

        self._cache_cursor = (self._cache_cursor+1) % self._cache_depth

        self._flush()

        self.refresh(*args, **kwargs)
