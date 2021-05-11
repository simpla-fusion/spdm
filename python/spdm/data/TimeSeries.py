import collections
from typing import Any, Generic, MutableMapping, Sequence, TypeVar, Union, Mapping

import numpy as np

from ..util.logger import logger
from .Node import Dict, List, _TIndex, _TKey, _TObject
from .AoS import AoS, SoA


class TimeSlice(Dict):
    r"""
        Time Slice
        - collection of property at same time point
        - collection of property interpolation at same time
    """
    __slots__ = ("_time")

    def __init__(self, *args, time: float = None,   **kwargs) -> None:
        Dict.__init__(self, *args, **kwargs)
        self._time = time or self["time"] or -np.inf

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
            return False
            # raise TypeError(type(other))


class TimeSeries(List[_TObject]):
    r"""
        Time Series
        - the collestion of propertis' time series . SOA (structure of array)
        - time series of the collection of properties  AOS (array of structure)
    """
    __slots__ = ("_time_step", "_time_start")

    def __init__(self, d: Union[Mapping, Sequence], *args, time_start=None, time_step=None,  **kwargs) -> None:
        if isinstance(d, collections.abc.Mapping):
            pass
        super().__init__(d, *args, **kwargs)
        self._time_start = time_start or 0.0
        self._time_step = time_step or 1.0

    @property
    def time(self) -> np.ndarray:
        return np.asarray([t_slice.time for t_slice in self])

    def last_time(self):
        if len(self) == 0:
            return self._time_start
        else:
            return float(self[-1].time)

    def next_time(self, dt=None):
        return self.last_time() + (dt or self._time_step)

    def __getitem__(self, k: _TIndex) -> _TObject:
        obj = super().__getitem__(k)

        if not self.__check_template__(obj.__class__):
            raise KeyError((k, obj))
        elif obj._time == -np.inf:
            n = len(self)
            obj._time = self._time_start+((k+n) % n)*self._time_step

        return obj

    def __setitem__(self, k: _TIndex, obj: Any) -> _TObject:
        return self.insert(k, obj)

    def insert(self,   *args,  **kwargs) -> _TObject:
        if len(args) > 0 and self.__check_template__(args[0].__class__):
            value = args[0]
        else:
            value = self.__new_child__(*args,  **kwargs)

        if value._time == -np.inf:
            value._time = self.next_time()

        return super().insert(value)

    def next(self, *args, time=None, **kwargs):
        return super().insert(self.__new_child__(*args, time=time or self.next_time(), **kwargs))

    def __call__(self, time: float = None) -> _TObject:
        r"""
            Time Interpolation of slices
        """
        logger.warning("NOTIMPLEMENTED!")
        return self[-1]
