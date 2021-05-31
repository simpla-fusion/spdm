import collections

from spdm.numlib import np
from spdm.data.Function import Function
from spdm.data.Node import Dict, Node, _TObject, _TKey
from spdm.data.Entry import Entry
from spdm.util.logger import logger


class Profiles(Dict[Node]):
    __slots__ = ("_axis",)

    def __init__(self,   *args, axis=None, ** kwargs):
        if isinstance(axis, int):
            axis = np.linspace(0, 1.0, axis)
        elif isinstance(axis, np.ndarray):
            axis = axis.view(np.ndarray)
        else:
            raise TypeError(type(axis))
        self._axis = axis
        super().__init__(*args, **kwargs)

    def __new_child__(self, value: _TObject, *args, parent=None,  **kwargs) -> Function:
        res = super().__new_child__(value, *args, parent=parent or self._parent, **kwargs)

        if value is None or (isinstance(value, Entry) and value.empty):
            value = None
        elif isinstance(value, Function):
            if value.x is not self._axis:
                value = Function(self._axis, np.asarray(value(self._axis)))
        elif callable(value):
            value = Function(self._axis, value)
        elif isinstance(value, np.ndarray) and len(value.shape) > 0:
            if value.shape != self._axis.shape:
                raise ValueError(f"The shape of arrays dismatch! {value.shape} !={self._axis.shape} ")
            value = Function(self._axis, value)
        return value
