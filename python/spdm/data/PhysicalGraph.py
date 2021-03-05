from .Quantity import Quantity
from .Graph import Graph
from .Node import _next_,_last_
from .Coordinates import Coordinates
import numpy as np


class PhysicalGraph(Graph):

    def __init__(self, *args, coordinates=None, **kwargs) -> None:
        if coordinates is not None:
            coordinates = coordinates if isinstance(coordinates, Coordinates) else Coordinates(coordinates)

        if coordinates is not None:
            self._coordinates = coordinates

        super().__init__(*args, coordinates=coordinates, **kwargs)

    def __getattr__(self, k):
        if k.startswith("_"):
            return super().__getattr__(k)
        else:
            return self.__getitem__(k)

    def __setattr__(self, k, v):
        if k.startswith("_"):
            super().__setattr__(k, v)
        else:
            self.__setitem__(k, v)

    def __delattr__(self, k):
        if k.startswith("_"):
            super().__delattr__(k)
        else:
            self.__delitem__(k)

    @property
    def coordinates(self):
        return getattr(self, "_coordinates", None) or getattr(self._value, "coordinates", None) or getattr(self._parent, "coordinates", None)

    def __pre_process__(self, value, *args, coordinates=None, **kwargs):
        # if not isinstance(value, (Quantity, collections.abc.Mapping, collections.abc.Sequence)) or isinstance(value, str):
        if isinstance(value, np.ndarray) and not isinstance(value, Quantity):
            value = Quantity(value, *args, coordinates=coordinates or self.coordinates, **kwargs)
        return value
