from __future__ import annotations

import collections.abc
import typing

import numpy as np

from ..utils.logger import logger
from ..utils.tags import _not_found_
from ..utils.tree_utils import merge_tree_recursive
from ..utils.typing import ArrayType, array_type, as_array
from .Entry import Entry
from .HTree import List
from .sp_property import SpDict, sp_property


class TimeSlice(SpDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    time: float = sp_property(unit="s", type="dynamic", default_value=0.0)  # type: ignore

    def refresh(self, *args,  **kwargs) -> TimeSlice:
        if len(args)+len(kwargs) > 0:
            super().update(*args, **kwargs)
        return self

    def advance(self, *args, dt: float | None = None, **kwargs) -> TimeSlice:
        if dt is None:
            if self._entry is not None:
                entry = self._entry.next()
                cache = None
            else:
                raise RuntimeError(f"Unkonwn dt={dt}")
        else:
            entry = None
            cache = {"time": self.time + dt}

        return self.__class__(cache, *args, entry=entry, parent=self._parent, **kwargs)


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

    def __init__(self, *args, start_slice: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_slice = start_slice or self._metadata.get("start_slice", None)

    @property
    def time(self) -> ArrayType:
        """时间序列"""
        return as_array([getattr(time_slice, "time", None) for time_slice in self._cache])

    @property
    def empty(self) -> bool: return self._cache is None or self._cache is _not_found_ or len(self._cache) == 0

    @property
    def current(self) -> _TSlice: return self[-1]

    # @property
    # def previous(self) ->_T: return self[-1]

    # @property
    # def next(self) ->_T: return self[1]

    def _find_by_time(self, time) -> int:
        """查找时间片所在的 entry 的位置"""

        idx = None
        for idx, t_slice in enumerate(self._cache):
            if isinstance(t_slice, TimeSlice):
                if time >= t_slice.time:
                    return idx

            elif isinstance(t_slice, dict):
                t = t_slice.get("time", None)

                if time >= t:
                    return idx

        time_coord = getattr(self, "_time_coord", None)

        if time_coord is None:
            time_coord = self._metadata.get("coordinate1", None)
            if time_coord is not None and self._entry is not None:
                time_coord = self._entry.child(f"../{time_coord}").fetch()
            self._time_coord = time_coord

        if isinstance(time_coord, np.ndarray):
            indices = np.where(time_coord <= time)[0]
            if len(indices) > 0:
                idx = indices[-1]
                if self._start_slice is None:
                    self._start_slice = idx
                return idx-self._start_slice

        if self._entry is not None and self._start_slice is not None:
            pos = self._start_slice
            while True:
                t_time = self._entry.child(f"{pos}/time").fetch()
                if t_time is _not_found_ or t_time is None:
                    break
                elif t_time >= time:
                    return pos-self._start_slice
                else:
                    pos += 1

        raise RuntimeError(f"Time {time} not found")

    def _pre_load(self, idx: int | None = None, time: float | None = None) -> int:
        """ load slices into cache, 返回最后一个slice在 cache中的位置
            NOTE: 不会触发 entry 读取数据，
        """

        if idx is not None:
            pass
        elif time is not None:
            idx = self._find_by_time(time)
        else:
            raise RuntimeError(f"{time} {idx}")

        length = len(self._cache)

        if idx < 0:
            idx += length

        if idx < 0:
            self._cache = [None]*(-idx)+self._cache

            if self._start_slice is None:
                self._start_slice = 0

            self._start_slice += idx+1

            if self._start_slice < 0:
                raise RuntimeError(f"Can not determine start_slice {self._start_slice}")

            idx = 0

        elif idx >= length:
            self._cache += [None]*(idx+1-length)

        return idx

    def __getitem__(self, idx: int) -> _TSlice:

        idx = self._pre_load(idx)

        value = self._cache[idx]

        if not isinstance(value, TimeSlice):

            entry = self._entry.child(self._start_slice+idx) if self._entry is not None else None

            value = self._as_child(value, None, entry=entry, parent=self._parent)

            self._cache[idx] = value  # type:ignore

        return value  # type:ignore

    def __setitem__(self, idx: int, value):
        idx = self._pre_load(idx)
        self._cache[idx] = value

    def refresh(self, *args, time: float | None = None, **kwargs) -> _TSlice:
        """
            更新 current 时间片状态。
            1. 若 time 非空，则将 current 指向新的 time slice.
            2. 调用 time_slice.refresh(*args,**kwargs)
        """

        if time is not None:
            idx = self._pre_load(time=time)
            self[idx].refresh(*args, **kwargs)
        else:
            self.current.refresh(*args, **kwargs)

        return self.current

    def advance(self, *args,  **kwargs) -> _TSlice:
        """
            由 current 时间片slice，推进出新的时间片（new slice）并追加在序列最后。

            如 dt is None , 则移动到 entry 中的下一个slice
        """

        self._cache.append(self.current.advance(*args,   **kwargs))  # type:ignore

        return self.current
