
import collections.abc
import typing

import numpy as np

from ..utils.logger import logger
from ..utils.tags import _not_found_
from .HTree import AoS, Dict, List
from .sp_property import SpDict, sp_property


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
        self._time = None

    @property
    def time(self) -> typing.List[float]:
        if self._time is None and self._parent is not None:
            self._time = self._parent.get(self._metadata.get("coordinate1", "time"), [0.0])
        else:
            self._time = [0.0]
        return self._time

    def __getitem__(self, index: int | slice | float) -> _T:

        if isinstance(index, int) and index < 0 and self.time is not None:
            new_key = len(self.time) + index
            if new_key < 0:
                raise KeyError(f"TimeSeries too short! length={len(self.time)} < {-index}")
            else:
                index = new_key
        elif isinstance(index, float):
            index = np.argmax(np.asarray(self.time) < index)
            logger.debug("TODO: interpolator two time slices!")

        return super().__getitem__(index)

    @property
    def previous(self) -> _T: return self[-2]

    @property
    def current(self) -> _T: return self[-1]

    def refresh(self,  *args, **kwargs) -> _T:
        """
            update the last time slice
        """
        if len(self) == 0:
            raise RuntimeError(f"TimeSeries is empty!")

        current_slice = self.current
        if hasattr(current_slice.__class__, "refresh"):
            current_slice.refresh(*args, **kwargs)
        elif len(args) == 1 and len(kwargs) == 0:
            current_slice.update(args[0])
        else:
            type_hint = self._type_hint(-1)
            new_obj = type_hint(*args, **kwargs, parent=self._parent)
            self[-1] = new_obj

        return self[-1]

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
            type_hint = self._type_hint()
            new_obj = type_hint(*args, time=time, **kwargs, parent=self._parent)

        self.insert(new_obj)

        return new_obj  # type: ignore
