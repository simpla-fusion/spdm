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

_T = typing.TypeVar("_T")


class TimeSlice(SpDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    time: float = sp_property(unit="s", type="dynamic", default_value=0.0)  # type: ignore

    def refresh(self, *args,  **kwargs) -> TimeSlice:

        if self._cache is None or self._cache is _not_found_:
            self._cache = kwargs

        if len(args) > 0 or len(kwargs) > 0:
            super().update(*args, **kwargs)

        return self

    def advance(self, *args, dt: float, **kwargs) -> TimeSlice:
        if self._entry is not None:
            entry = self._entry.next()
        else:
            entry = None

        next_one = self.__class__({"time": self.time+dt}, *args, entry=entry, parent=self._parent, **kwargs)

        return next_one


class TimeSeriesAoS(List[_T]):
    """ A series of time slices .
        TODO:
          1. 缓存时间片，避免重复创建，减少内存占用
          2. 缓存应循环使用
          3. cache 数据自动写入 entry 落盘


    """

    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        # 当前 slice 在 cache 中的位置
        self._cache_cursor: int = None    # type:ignore

    @property
    def time(self) -> ArrayType:
        return as_array([getattr(time_slice, "time", None) for time_slice in self._cache])

    @property
    def previous(self) -> TimeSlice | _T: return self[-1]

    @property
    def current(self) -> TimeSlice | _T: return self[0]

    @property
    def next(self) -> TimeSlice | _T: return self[1]

    def _extend_cache(self, rel_index: int = 0) -> int:

        if self._cache is None or self._cache is _not_found_:
            self._cache = [None]
            self._cache_cursor = 0
        elif len(self._cache) == 0 or self._cache_cursor is None:
            self._cache.append(None)
            self._cache_cursor = 0

        num = len(self._cache)

        cache_pos = self._cache_cursor + rel_index

        # 扩展 cache
        if cache_pos < 0:
            self._cache = [None]*(-cache_pos)+self._cache
            self._cache_cursor -= cache_pos
            cache_pos = 0
        elif cache_pos >= num:
            self._cache += [None]*(cache_pos-num+1)

        return cache_pos

    def _find_slice(self,  *args, cache=None, hint_time=None, hint_index=None, **kwargs) -> TimeSlice | _T:
        entry = None
        entry_pos = None

        value = self._as_child(cache, entry_pos, *args, entry=entry, parent=self._parent, **kwargs)

        return value

    def __getitem__(self, idx: int) -> TimeSlice | _T:
        if not isinstance(idx, int):
            raise NotImplementedError(f"{idx}")

        pos = self._extend_cache(idx)

        value = self._cache[pos]

        if not isinstance(value, TimeSlice):
            value = self._find_slice(None, cache=value, hint_index=idx)
            self._cache[pos] = value  # type:ignore

        return value

    def __setitem__(self, idx: int, value): self._cache[self._extend_cache(idx)] = value

    def refresh(self, *args, **kwargs) -> TimeSlice | _T:
        if self._cache is None or self._cache is _not_found_ or len(self._cache) == 0:
            pos = self._extend_cache()
            self._cache[pos] = self._find_slice(kwargs.pop("time", 0.0))  # type:ignore

        return self.current.refresh(*args, **kwargs)  # type:ignore

    def advance(self, *args, **kwargs) -> TimeSlice | _T:

        if self._cache is None or self._cache is _not_found_ or len(self._cache) == 0:
            pos = self._extend_cache()
            self._cache[pos] = self._find_slice(kwargs.pop("time", 0.0))  # type:ignore

        else:
            self[1] = self.current.advance(*args, **kwargs)  # type:ignore
            self._cache_cursor += 1

        return self.current
