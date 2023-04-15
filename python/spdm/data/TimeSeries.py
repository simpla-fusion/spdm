import typing
import collections.abc
import numpy as np

from .Function import Function


class TimeSeries(Function):

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], collections.abc.Mapping):
            data = args[0].get("data", None)
            time = args[0].get("time", None)
        elif len(args) == 2:
            time = args[0]
            data = args[1]
        else:
            raise ValueError("Can not find 'time' and 'data'!")

        super().__init__(time, data,  **kwargs)

    @property
    def data(self) -> np.ndarray:
        return self.__array__()

    @property
    def time(self) -> np.ndarray:
        return self._x_axis
