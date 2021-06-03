import collections
from typing import Generic

from spdm.numlib import np
from spdm.data.Function import Function
from spdm.data.Node import Dict, Node, _TObject, _TKey
from spdm.data.Entry import Entry
from spdm.util.logger import logger
from spdm.util.utilities import _not_found_


class Profiles(Dict[_TObject]):
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

        self.__new_child__ = lambda value, parent=self: Function(parent._axis, value) if isinstance(
            value, np.ndarray) and value.shape == parent._axis.shape else value
