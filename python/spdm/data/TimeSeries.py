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

    def __init__(self, *args, time=None, dt=0.0,  **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(time, np.ndarray):
            time = time.tolist()
        elif time is None:
            time = []
        self._time = time
        self._dt = dt

    @property
    def time(self) -> _TTimeSeries:
        return self._time

    @property
    def last_time(self):
        return self._time[-1] if len(self._time) > 0 else 0.0

    @property
    def new_time(self):
        return self.last_time + self._dt

    def __new_child__(self,  *args,  time=None, **kwargs):
        return super().__new_child__(*args, time=time or self.new_time, **kwargs)

    def insert(self, d: _TObject, *args, time: _TTime = None, **kwargs) -> _TObject:
        if time < self.last_time:
            raise NotImplementedError(f"{time} > {self.last_time}")
        time = time or self.new_time
        obj = self.__new_child__(d, *args, time=time, **kwargs)
        self._time.append(getattr(obj, time, None) or time)
        super().insert(obj)
        return obj

    def get_slice(self, time: _TTime = None) -> TimeSlice[_TObject]:
        """
           Time Interpolation 
        """
        return TimeSlice[_TObject](self, time)

    def __call__(self, time: _TTime) -> TimeSlice[_TObject]:

        return self.get_slice(time)

    def __getattr__(self, k) -> Any:
        return getattr(self.get_slice(self.last_time), k)
