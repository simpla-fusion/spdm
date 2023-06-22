
import collections.abc
import typing

from spdm.data.Dict import Dict
from spdm.data.List import List, AoS
from spdm.data.Node import Node
from spdm.data.sp_property import SpDict, sp_property
from spdm.utils.logger import logger
import numpy as np


class TimeSlice(SpDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    time: float = sp_property(unit='s', type='dynamic', default_value=0.0)  # type: ignore


_T = typing.TypeVar("_T")


class TimeSeriesAoS(AoS[_T]):
    """
        A series of time slices, each time slice is a state of the system at a given time.
        Each slice is a dict .
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._time = None

    @property
    def time(self) -> list[float]:
        if self._time is None:
            self._time = self._parent.get(self._metadata.get("coordinate1", "time"), None)
        return self._time

    def __getitem__(self, index) -> _T:
        time = getattr(self, "time", None)
        if isinstance(index, int) and index < 0 and time is not None:
            new_key = len(time) + index
            if new_key < 0:
                raise KeyError(f"TimeSeries too short! length={len(time)} < {-index}")
            else:
                index = new_key
        elif isinstance(index, float):
            index = np.argmax(np.asarray(time) < index)
            logger.debug("TODO: interpolator two time slices!")

        return super().__getitem__(index)

    @property
    def previous(self) -> _T: return self[-2]

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

    def advance(self, *args, time: float = ..., **kwargs) -> _T:
        self.time.append(time)
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
