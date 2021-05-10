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

    # def __new_child__(self, value, *args, parent=None, time=None, **kwargs):
    #     return super().__new_child__(value, *args, parent=parent, time=time, **kwargs)

    def __getitem__(self, k: _TIndex) -> _TObject:
        obj = super().__getitem__(k)
        if obj._time is None or obj._time == None:
            obj._time = self._time_start+k*self._time_step
        return obj

    def __setitem__(self, k: _TIndex, obj: Any) -> _TObject:
        if isinstance(k, int):
            logger.warning("FIXME: Untested features!")
            obj = super().__new_child__(obj, parent=self)
            if obj._time is None or obj._time == None:
                obj._time = self._time_start+k*self._time_step
            super().__setitem__(k, obj)
        elif isinstance(k, float):
            self.insert(super().__new_child__(obj, time=k, parent=self))
        else:
            super().__setitem__(k, obj)

    def last_time_step(self):
        return 0.0 if len(self) == 0 else self[-1].time

    def next_time_step(self, dt=None):
        return self.last_time_step() + (dt or self._time_step)

    def insert_by_order(self, obj, *args, **kwargs) -> _TimeSlice:
        if not hasattr(obj, "_time"):
            obj = self.__new_child__(obj, *args,  **kwargs)
            if obj._time == None:
                obj._time = self.next_time_step()
        self.insert_by_order(self, obj)
        return obj

    @property
    def last_time_slice(self) -> _TimeSlice:
        if len(self) == 0:
            raise IndexError(f"Empty list")
        return self[-1]

    @property
    def next_time_slice(self) -> _TimeSlice:
        return self.insert(self.last_time_slice.__duplicate__(time=self.next_time_step()))

    def get_slice(self, time: float = None) -> _TimeSlice:
        """
           Time Interpolation 
        """
        return NotImplemented

    def __call__(self, time: float = None) -> _TimeSlice:
        if time is None:
            return self[-1]
        else:
            return self.get_slice(time)
