import collections
from typing import Generic

import numpy as np
from spdm.data.Function import Function
from spdm.data.Node import Dict, Node, _TObject, _TKey
from spdm.data.Entry import Entry
from spdm.util.logger import logger
from spdm.util.utilities import _not_found_


class Profiles(Dict[_TObject]):
    __slots__ = ("_axis",)

    def __init__(self,  d, /, axis: np.ndarray, ** kwargs):
        def new_child(value, _axis=axis):
            return Function(_axis, value) if isinstance(value, np.ndarray) and value.shape == _axis.shape else value
        super().__init__(d, new_child=new_child, **kwargs)
        self._axis = axis

    @property
    def axis(self) -> np.ndarray:
        return self._axis
