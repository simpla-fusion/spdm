import numpy as np
from .Function import Function
from ..tags import _not_found_


class Signal(Function):

    def __init__(self, time: np.ndarray, data: np.ndarray = None, **kwargs):
        if data is None:
            data = time.get("data", _not_found_)
            time = time.get("time", _not_found_)
            if time is _not_found_:
                raise ValueError(f"Can not find 'time'! {time}")
            elif data is _not_found_:
                raise ValueError("Can not find 'data'!")
        super().__init__(time, data,  **kwargs)

    @property
    def data(self) -> np.ndarray:
        return self.__array__()

    @property
    def time(self) -> np.ndarray:
        return self._x_axis
