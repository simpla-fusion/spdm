from typing import Any

from spdm.utils.tags import _not_found_

from .Field import Field
from .Function import Function
from .List import AoS, List
from .Node import Node
from .Signal import Signal, SignalND
from .sp_property import SpDict, sp_property
from .TimeSeries import TimeSeriesAoS, TimeSlice
from ..utils.Pluggable import Pluggable
import typing


class Actor(SpDict, Pluggable):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    time: list = sp_property(type="dynamic", units="s", default_value=[])

    def time_slice(self, time: int | float | typing.List[float]) -> TimeSlice | typing.List[TimeSlice]:
        raise NotImplementedError()

    def advance(self,  *args, time: float, ** kwargs):
        self.time.append(time)

    def update(self,  *args,  ** kwargs):
        super().update(*args, **kwargs)
