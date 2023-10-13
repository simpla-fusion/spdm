from __future__ import annotations

import collections.abc
import typing

import numpy as np

from ..utils.logger import logger
from ..utils.tags import _not_found_
from ..utils.tree_utils import merge_tree_recursive
from ..utils.typing import ArrayType, array_type, as_array
from .Entry import Entry
from .HTree import List, HTree
from .sp_property import SpTree, sp_property


class TimeSlice(SpTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    time: float = sp_property(unit="s",   default_value=0.0)  # type: ignore

    def refresh(self, *args,  **kwargs) -> TimeSlice:
        if len(args) == 1 and isinstance(args[0], dict):
            super().update(args[0])
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

        return self.__class__(cache, *args, _entry=entry, _parent=self._parent, **kwargs)


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

    def dump(self, entry: Entry, **kwargs) -> None:
        """ 将数据写入 entry """
        entry.insert([{}]*len(self._cache))
        for idx, value in enumerate(self._cache):
            if isinstance(value, HTree):
                value.dump(entry.child(idx), **kwargs)
            else:
                entry.child(idx).insert(value)

    @property
    def time(self) -> ArrayType:
        """时间序列"""
        return as_array([getattr(time_slice, "time", None) for time_slice in self._cache])

    @property
    def dt(self) -> float: return self._metadata.get("dt", 0.1)

    @property
    def empty(self) -> bool: return self._cache is None or self._cache is _not_found_ or len(self._cache) == 0

    @property
    def current(self) -> _TSlice:
        if not self.empty:
            pass
        elif self._entry is not None:
            self._cache.append(self._entry.child(self._start_slice or 0))
        else:
            raise RuntimeError(f"Can not get current slice! {self._start_slice}")

        return self[-1]

    def _find_slice_index_by_time(self, time) -> typing.Tuple[int, float]:

        time_coord = getattr(self, "_time_coord", None)

        if time_coord is None:

            time_coord = self._metadata.get("coordinate1", None)

            if time_coord is not None and self._entry is not None:
                self._time_coord = self._entry.child(f"../{time_coord}").fetch()
            else:
                self._time_coord = _not_found_

            time_coord = self._time_coord

        pos: int = None

        if isinstance(time_coord, np.ndarray):

            indices = np.where(time_coord <= time)[0]

            if len(indices) > 0:
                pos = indices[-1]
                time = time_coord[pos]

        elif self._entry is not None and self._start_slice is not None:
            pos = self._start_slice
            while True:
                t_time = self._entry.child(f"{pos}/time").fetch(default_value=_not_found_)
                if t_time is _not_found_ or t_time is None:
                    break
                elif t_time >= time:
                    time = t_time
                    break
                else:
                    pos += 1
        else:
            pos = None
            # raise RuntimeError(f"Unkonwn time_coord={time_coord} start_slice={self._start_slice} time={time}")

        return pos, time

    def _extend_cache(self, idx):

        if self._cache is None or self._cache is _not_found_:
            self._cache = []

        if idx < 0:
            idx += len(self._cache)

            if idx == -1:
                idx = 0
            elif idx < 0:
                raise IndexError(
                    f"Out of range ! {idx} cache length={len(self._cache)} start index={self._start_slice}")

        if self._start_slice is None:
            self._start_slice = idx
        elif idx >= 0:
            idx -= self._start_slice
        else:
            idx += len(self._cache)

        if idx < 0:
            self._start_slice += idx
            self._cache = [None]*(-idx)+self._cache
            idx = 0
        elif idx >= len(self._cache):
            self._cache += [None]*(idx-len(self._cache)+1)

    def __getitem__(self, idx: int) -> _TSlice:

        self._extend_cache(idx)

        value = self._cache[idx]
        entry = None

        if not isinstance(value, TimeSlice):

            if isinstance(value, Entry):
                entry = value
                value = None

            elif self._entry is None or self._entry is _not_found_:
                pass

            elif isinstance(value, dict) and "time" in value:

                time = value.get("time")

                pos, t_time = self._find_slice_index_by_time(time)

                if pos is None:
                    # raise RuntimeError(f"Can not find slice! {value}")
                    entry = None
                else:
                    if not np.isclose(t_time, time):
                        logger.warning(f"Found closest slice. {time}->{t_time}")
                        value["time"] = t_time

                    entry = self._entry.child(pos)
                    if idx == 0:
                        self._start_slice = pos

            elif self._start_slice is not None and value is None:
                entry = self._entry.child(self._start_slice+idx)

            else:
                raise TypeError(f"Unknown type {type(value)}")

            value = self._as_child(value, None, _entry=entry, _parent=self._parent)

            self._cache[idx] = value

        return value  # type:ignore

    def __setitem__(self, idx: int, value):
        self._extend_cache(idx)
        self._cache[idx] = value

    def refresh(self, *args, time: float | None = None, **kwargs) -> _TSlice:
        """
            更新 current 时间片状态。
            1. 若 time 非空，则将 current 指向新的 time slice.
            2. 调用 time_slice.refresh(*args,**kwargs)
        """

        if time is None:
            self.current.refresh(*args, **kwargs)

        elif not self.empty and time < self.current.time:
            # 在序列中间插入 slice
            raise NotImplementedError(f"TODO: insert slice! {time}<  {self.current.time}")

        else:
            if self.empty or time > self.current.time:  # 插入新的 slice
                self._cache.append({"time": time})

            self.current.refresh(*args, **kwargs)

        return self.current

    def advance(self, *args, time: float | None = None, **kwargs) -> _TSlice:
        """
            由 current 时间片slice，推进出新的时间片（new slice）并追加在序列最后。

            如 dt is None , 则移动到 entry 中的下一个slice
        """
        if self.empty:
            self.refresh(*args, time=time, **kwargs)

        elif len(args)+len(kwargs) > 0:
            self._cache.append(self.current.advance(*args, time=time, **kwargs))

        elif time is None:
            if self.current._entry is None:
                raise RuntimeError(f"Can not find next slice! {self.current}")

            self._cache.append(self.current._entry.next())

        elif time < self.current.time:
            raise RuntimeError(f"Can not insert slice! {time}<{self.current.time}")

        else:
            self._cache.append({"time": time})

        return self.current
