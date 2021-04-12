import collections
import functools

import numpy as np

from .Graph import Graph

from ..util.logger import logger
from .AttributeTree import as_attribute_tree


@as_attribute_tree
class PhysicalGraph(Graph):
    r"""
       "PhysicalGraph" is a set of "quantities" (nodes) with internal mutual constraints (edges).

    """

    def __init__(self,   *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __update__(self, *args, **kwargs):
        super().__update__(*args, **kwargs)

    def __new_child__(self, *args, parent=None, **kwargs):
        return PhysicalGraph(*args,  parent=parent or self, **kwargs)

    # def __pre_process__(self, value, *args,   **kwargs):
    #     # if isinstance
    #     # if isinstance(value, np.ndarray) and not isinstance(value, Quantity):
    #     #     value = Quantity(value, *args, coordinates=coordinates or self.coordinates, **kwargs)
    #     return super().__pre_process__(value, *args, **kwargs)

    # def __post_process__(self, value, *args, **kwargs):
    #     return super().__post_process__(value, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        value = self.__value__
        if callable(value):
            return value(*args, **kwargs)
        else:
            return value
