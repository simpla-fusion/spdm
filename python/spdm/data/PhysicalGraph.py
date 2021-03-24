import functools

import numpy as np

from .Graph import Graph
from .Quantity import Quantity
from ..util.utilities import try_get, try_put
from ..util.logger import logger
from .Entry import Entry


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

    def __new_node__(self, *args, parent=None, **kwargs):
        return PhysicalGraph(*args,  parent=parent or self, **kwargs)

    def __getattr__(self, k):
        if k.startswith("_"):
            return self.__dict__[k]
        else:
            # res = getattr(self.__class__, k, None)
            # if res is None:
            #     res = self.__getitem__(k)
            # elif isinstance(res, property):
            #     res = getattr(res, "fget")(self)
            # elif isinstance(res, functools.cached_property):
            #     res = res.__get__(self)
            res = try_get(self, k, None)
            if res is None:
                res = self.__class__(Entry(self._data, prefix=[k]))
            return res

    def __setattr__(self, k, v):
        if k.startswith("_"):
            super().__setattr__(k, v)
        else:
            try_put(self, k, v)
            # res = getattr(self.__class__, k, None)
            # if res is None:
            #     self.__setitem__(k, v)
            # elif isinstance(res, property):
            #     res.fset(self, k, v)
            # else:
            #     raise AttributeError(f"Can not set attribute {k}!")

    def __delattr__(self, k):
        if k.startswith("_"):
            super().__delattr__(k)
        else:
            res = getattr(self.__class__, k, None)
            if res is None:
                self.__delitem__(k)
            elif isinstance(res, property):
                res.fdel(self, k)
            else:
                raise AttributeError(f"Can not delete attribute {k}!")

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
