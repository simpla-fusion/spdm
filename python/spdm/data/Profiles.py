import collections

from spdm.numlib import np
from spdm.data.Function import Function
from spdm.data.Node import Dict, Node, _TObject, _TKey
from spdm.data.Entry import Entry
from spdm.util.logger import logger
from spdm.util.utilities import _not_found_


class Profiles(Dict[Node]):
    __slots__ = ("_axis",)

    def __init__(self,   *args, axis=None, ** kwargs):

        super().__init__(*args, **kwargs)
        if isinstance(axis, int):
            axis = np.linspace(0, 1.0, axis)
        elif isinstance(axis, np.ndarray):
            axis = axis.view(np.ndarray)
        else:
            axis = getattr(self._parent, "_axis", None)
        self._axis = axis

    def __new_child__(self, value: _TObject, *args, parent=None,   **kwargs) -> Function:
        if isinstance(value, np.ndarray) and value.shape == self._axis.shape:
            value = Function(self._axis, value)
        elif isinstance(value, Entry):
            value = super().__new_child__(value, *args, parent=parent or self._parent, **kwargs)

        return value
