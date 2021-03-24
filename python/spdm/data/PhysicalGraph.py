import functools

import numpy as np

from .Graph import Graph
from .Quantity import Quantity
from ..util.logger import logger
from .AttributeTree import as_attribute_tree


@as_attribute_tree
class PhysicalGraph(Graph):
    r"""
       "PhysicalGraph" is a set of "quantities" (nodes) with internal mutual constraints (edges).

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._changed = True

    @property
    def __changed__(self):
        return self._changed

    def __update__(self, *args, **kwargs):
        super().__update__(*args, **kwargs)
        self._changed = True

    def __new_child__(self, *args, parent=None, **kwargs):
        return PhysicalGraph(*args,  parent=parent or self, **kwargs)

    def __pre_process__(self, value, *args, coordinates=None, **kwargs):
        if isinstance(value, np.ndarray) and not isinstance(value, Quantity):
            value = Quantity(value, *args, coordinates=coordinates or self.coordinates, **kwargs)
        return value

    def __postprocess__(self, value, *args, **kwargs):
        return super().__post_process__(value, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        value = self.__value__
        if callable(value):
            return value(*args, **kwargs)
        else:
            return value
