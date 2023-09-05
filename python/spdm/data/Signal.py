import typing

import numpy as np
from spdm.data.Entry import Entry
from spdm.data.HTree import HTree

from ..utils.logger import logger
from ..utils.typing import ArrayType
from .Field import Field
from .Function import Function
from .HTree import HTree
from .sp_property import SpDict, sp_property

_T = typing.TypeVar("_T")


class Signal(SpDict[_T]):
    """Signal with its time base    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._func = None

    data: np.ndarray = sp_property(type="dynamic   ")

    time: np.ndarray = sp_property(units="s", type="dynamic   ")

    def __call__(self, t: float) -> float:
        if self._func is None:
            self._func = Function(self.data, self.time)
        return self._func(t)


class SignalND(Signal[_T]):
    pass
