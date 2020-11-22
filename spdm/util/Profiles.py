
import collections
from copy import copy
import inspect

import numpy as np
import pprint
from .AttributeTree import AttributeTree
from .Interpolate import Interpolate1D, Interpolate2D, InterpolateND
# from .logger import logger


class Profiles(AttributeTree):
    """ Collection of '''profile'''s with common '''dimension'''.

        dims likes [0,1,2,3,4,5]    or [  [0,1,2,3,4,5]  ]
    """
    @staticmethod
    def create_dims(d):
        if isinstance(d, list):
            dims = [Profiles.create_dims(dim) for dim in d]
            shape = [len(v) for v in dims]
            return shape, dims
        elif isinstance(d, int):
            return [d], np.linspace(0, 1, d)
        elif isinstance(d, tuple):
            x0, x1, num = d
            return [num], np.linspace(x0, x1, num)
        elif isinstance(d, np.ndarray):
            return d.shape, d
        else:
            raise TypeError(f"Illegal dimension type! {type(d)}")

    def __init__(self,  dims=None, *args, interpolator=None, ** kwargs):
        if dims is None:
            dims = 129
        self.__shape__, self.__dimensions__ = Profiles.create_dims(dims)

        if len(self.__shape__) == 1:
            self.__mgrid__ = [self.__dimensions__]
            self.__interpolator__ = interpolator or Interpolate1D
        elif len(self.__dime__shape__nsions__) == 2:
            self.__mgrid__ = np.meshgrid(self.__dimensions__[0], self.__dimensions__[1], indexing='ij')
            self.__interpolator__ = interpolator or Interpolate2D
        else:
            raise NotImplementedError()

        super().__init__(*args, default_factory=lambda _s=self.__shape__: np.zeros(shape=_s), ** kwargs)

    def copy(self):
        return Profiles(self.__dimensions__, interpolator=self.__interpolator__)

    @property
    def dimensions(self):
        return self.__dimensions__

    @property
    def grid(self):
        return self.__mgrid__

    def __getitem__(self, key):
        if self.__class__ is not Profiles:
            obj = getattr(self.__class__, key, None)

        if isinstance(obj, property):
            return getattr(obj, "fget")(self)
        elif inspect.isfunction(obj):
            return lambda *args, _self=self, _fun=obj, **kwargs, : _fun(_self, *args, **kwargs)
        else:
            obj = super().__getitem__(key)
            return self.__interpolator__(*self.grid, obj)

    def __setitem__(self, key, value):
        if value is None:
            value = np.zeros(shape=self.dimensions.shape)

        if isinstance(value, collections.abc.Mapping):
            self.__data__.setdefault(key, self.copy()).__update__(value)
        else:
            super().__setitem__(key, value)


if __name__ == "__main__":
    a = Profiles(np.linspace(0, 1, 10))
    a.b = {}
    a.b.c = 5

    # d = a.b.f.__push_back__()
    # d.text = "hellow world"
    pprint.pprint(a)
    pprint.pprint(a.b.d.shape)
    pprint.pprint(a.dimensions)

    # pprint.pprint(a.b.c.dimensions)
    # a.dimensions.fill(0)
    # pprint.pprint(a.dimensions)

    # pprint.pprint(a.b.c.dimensions)
    # pprint.pprint(a.b.e.__dimensions__)
