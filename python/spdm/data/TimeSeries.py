import collections
from typing import (Any, Generic, Sequence, TypeVar)
import typing
from functools import cached_property
import numpy as np
from ..util.generic_template import get_template_args
from ..util.logger import logger
from .AttributeTree import AttributeTree, as_attribute_tree
from .Function import Function
from .Node import Dict, List,   _TIndex, _TObject, _TKey
_TTime = float
_TTimeSeries = Sequence[float]


class TimeSequence(Sequence[float]):
    def __init__(self, time, *args, dt=None, **kwargs) -> None:
        super().__init__()
        if isinstance(time, np.ndarray):
            time = time.tolist()
        elif time == None:
            time = []
        elif not isinstance(time, collections.abc.MutableSequence):
            time = [time]
        self._time = time
        self._dt = dt

    @property
    def last(self) -> float:
        return self._time[-1] if isinstance(self._time, collections.abc.Sequence) and len(self._time) > 0 else 0.0

    @property
    def next(self) -> float:
        return self.last_time + self._dt

    def append(self, time):
        if time < self.last_time:
            raise NotImplementedError(f"{time} > {self.last_time}")

        self._data.append(time)

        return time

    def insert(self, time):
        return self._data.append(time)

    def __array__(self) -> np.ndarray:
        return np.asarray(self._data)

    def __getitem__(self, idx):
        return self._time[idx]

    def __len__(self) -> int:
        return len(self._time)


class TimeSlice(Generic[_TObject]):
    r"""
        Time Slice
        - collection of property at same time point
        - collection of property interpolation at same time
    """
    __slots__ = ("_time", "_series")

    def __init__(self, series: List[_TObject], time: float, *args, **kwargs) -> None:
        self._series = series
        self._time = time

    @property
    def time(self) -> float:
        return self._time


class TimeSeries(List[_TObject]):
    r"""
        Time Series
        - the collestion of propertis' time series . SOA (structure of array)
        - time series of the collection of properties  AOS (array of structure)
    """
    __slots__ = ("_time", "_dt")

    def __init__(self, *args, time=None, dt=None,  **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._time = TimeSequence(time, dt=dt)
        self._dt = dt or 0.0

    @property
    def time(self) -> TimeSequence:
        return self._time

    def __new_child__(self,  *args,    **kwargs):
        return super().__new_child__(*args,  **kwargs)

    def insert(self, d: _TObject, *args, time: _TTime = None, **kwargs) -> _TObject:
        obj = self.__new_child__(d, *args, time=self._time.append(time), **kwargs)
        super().insert(obj)
        return obj

    def get_slice(self, time: _TTime = None) -> TimeSlice[_TObject]:
        """
           Time Interpolation 
        """
        return TimeSlice[_TObject](self, time)

    def __call__(self, time: _TTime) -> TimeSlice[_TObject]:
        return self.get_slice(time)
