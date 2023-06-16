
import collections.abc
import typing

from spdm.data.Dict import Dict
from spdm.data.List import List, AoS
from spdm.data.Node import Node
from spdm.data.sp_property import SpDict, sp_property
from spdm.utils.logger import logger


class TimeSlice(SpDict):
    def __init__(self, *args, time=None, **kwargs):
        super().__init__(*args, **kwargs)
        if time is not None:
            self._cache["time"] = time

    time: float = sp_property(unit='s', type='dynamic', default_value=0.0)  # type: ignore


_T = typing.TypeVar("_T")


class TimeSeriesAoS(AoS[_T]):
    """
        A series of time slices, each time slice is a state of the system at a given time.
        Each slice is a dict .
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._time = self._parent.time

    def time_slice(self, time: int | slice | float | typing.Sequence[float]) -> TimeSlice | typing.List[TimeSlice]:
        if isinstance(time, (int, slice)):
            return self[time]
        elif isinstance(time, float):
            prev_node = None
            for next_node in self:
                next_time = getattr(next_node, "time", None)
                if next_time is not None and next_time > time:
                    break
                else:
                    prev_node = next_node
            else:                
                if prev_node is not None:
                    return prev_node
                
            raise NotImplementedError("TODO: find the nearest time slice or interpolate two time slices")
        
        elif isinstance(time, collections.abc.Generator):
            return [self.time_slice(t) for t in time]
        else:
            raise TypeError(f"{type(time)}")

    @property
    def current(self) -> _T: return self[-1]

    def update(self,  *args, **kwargs) -> _T:
        """
            update the last time slice
        """
        if len(self) == 0:
            raise RuntimeError(f"TimeSeries is empty!")

        if len(args) > 0 and isinstance(args[0], TimeSlice):
            new_obj = args[0]
            new_obj._parent = self._parent
            self[-1] = new_obj
        elif len(args) > 0 or len(kwargs) > 0:
            type_hint = self.__type_hint__()
            new_obj = type_hint(*args, **kwargs, parent=self._parent)
            self[-1] = new_obj
        else:
            new_obj = self[-1]

        return new_obj

    def advance(self, *args, time=None, **kwargs) -> _T:
        if isinstance(self._default_value, collections.abc.Mapping):
            kwargs.setdefault("default_value", self._default_value)

        if len(args) > 0 and isinstance(args[0], TimeSlice):
            new_obj = args[0]
            new_obj._parent = self._parent
            if time is not None:
                new_obj["time"] = time
        else:
            type_hint = self.__type_hint__()
            new_obj = type_hint(*args, time=time, **kwargs, parent=self._parent)

        self.append(new_obj)

        return new_obj  # type: ignore
