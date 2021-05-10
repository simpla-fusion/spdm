import collections
from typing import Any, Generic, Sequence, TypeVar

import numpy as np

from ..util.logger import logger
from .Node import Dict, List, _TIndex, _TKey, _TObject


class TimeSlice(Dict):
    r"""
        Time Slice
        - collection of property at same time point
        - collection of property interpolation at same time
    """
    __slots__ = ("_time")

    def __init__(self,   *args, time: float = None,   **kwargs) -> None:
        Dict.__init__(self, *args, **kwargs)
        self._time = time or self["time"]
        if isinstance(self._time, str):
            self._time = str(self._time)

    @property
    def time(self) -> float:
        return self._time

    def __duplicate__(self, time=None, parent=None):
        return self.__class__(time=time or self._time, parent=parent or self._parent)

    def __lt__(self, other):
        if isinstance(other, float):
            return self._time < other
        elif isinstance(other, TimeSlice):
            return self._time < other._time
        else:
            raise TypeError(type(other))


_TimeSlice = TypeVar("_TimeSlice", TimeSlice, Any)


class TimeSeries(List[_TimeSlice]):
    r"""
        Time Series
        - the collestion of propertis' time series . SOA (structure of array)
        - time series of the collection of properties  AOS (array of structure)
    """
    __slots__ = ("_time_step", "_time_start")

    def __init__(self, *args, time_start=None, time_step=None,  **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._time_start = time_start or 0.0
        self._time_step = time_step or 1.0

    @property
    def time(self) -> np.ndarray:
        return np.asarray([t_slice.time for t_slice in self])

    def last_time_step(self):
        return 0.0 if len(self) == 0 else self[-1].time

    def next_time_step(self, dt=None):
        return self.last_time_step() + (dt or self._time_step)

    def __getitem__(self, k: _TIndex) -> _TObject:
        obj = super().__getitem__(k)
        if obj._time == None:
            obj._time = self._time_start+k*self._time_step
        return obj

    def __setitem__(self, k: _TIndex, obj: Any) -> _TObject:
        return self.insert(k, obj)

    def insert(self, first, second=None) -> _TimeSlice:
        if second is not None:
            time = first
            value = second
        else:
            time = None
            value = first

        if not hasattr(value, "_time"):
            value = self.__new_child__(value)

        if isinstance(time, float):
            value._time = time
        elif isinstance(time, int):
            value._time = self._time_start+time*self._time_step
        elif value._time == None:
            value._time = self.next_time_step()

        return super().insert(value)

    def __call__(self, time: float = None) -> _TimeSlice:
        r"""
            Time Interpolation of slices
        """
        return NotImplemented
