
import typing

from spdm.data.Dict import Dict
from spdm.data.List import List
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property, SpDict
from spdm.utils.logger import logger


class TimeSlice(SpDict):
    time: float = sp_property(unit='s', type='dynamic', default_value=0.0)


_T = typing.TypeVar("_T")


class TimeSeriesAoS(List[TimeSlice]):
    """
        A series of time slices, each time slice is a state of the system at a given time.
        Each slice is a dict .
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def current(self) -> TimeSlice: return self[-1]

    def update(self, d=None, *args, **kwargs) -> TimeSlice:
        """
            update the last time slice, base on profiles_2d[-1].psi
        """
        if len(self) == 0:
            raise RuntimeError(f"TimeSeries is empty!")

        if hasattr(self[-1], "update"):
            self[-1].update(d, *args, **kwargs)
        elif isinstance(d, TimeSlice):
            d._parent = self._parent
            self[-1] = d
        elif d is not None:
            type_hint = self.__type_hint__()
            self[-1] = type_hint(d, *args, **kwargs, parent=self._parent)

        return self[-1]

    def advance(self, d, *args,  **kwargs) -> TimeSlice:

        if len(self) > 0 and hasattr(self[-1], "advance"):
            new_obj = self[-1].advance(d, *args, **kwargs)
        elif isinstance(d, TimeSlice):
            new_obj = d
            new_obj._parent = self._parent
        else:
            type_hint = self.__type_hint__()
            new_obj = type_hint(d, *args, **kwargs, parent=self._parent)
        self.append(new_obj)
        return d
