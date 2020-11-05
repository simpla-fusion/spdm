
import collections
from copy import copy
import inspect

import numpy as np
import pprint
from .AttributeTree import AttributeTree
# from .logger import logger


class Profiles(AttributeTree):
    """ Collection of '''profile'''s with common '''dimension'''.

        dims likes [0,1,2,3,4,5]    or [  [0,1,2,3,4,5]  ]
    """

    def __init__(self,  dims, *args, interpolator=None, ** kwargs):
        self.__interpolator__ = interpolator or (lambda x0, y0, x1: x1)
        self.__dimensions__ = dims if isinstance(dims, np.ndarray) else np.array(dims)
        super().__init__(*args, default_factory=lambda _s=self.__dimensions__.shape: np.zeros(shape=_s), ** kwargs)

    def copy(self):
        return Profiles(self.__dimensions__, interpolator=self.__interpolator__)

    @property
    def dimensions(self):
        return self.__dimensions__

    def __getitem__(self, key):
        if self.__class__ is not Profiles:
            obj = getattr(self.__class__, key, None)
            if isinstance(obj, property):
                return getattr(obj, "fget")(self)
            elif inspect.isfunction(obj):
                return lambda *args, _self=self, _fun=obj, **kwargs, : _fun(_self, *args, **kwargs)
            else:
                obj = super().__getitem__(key)
                return lambda *args,  _obj = obj, _dims=self.__dimensions__, _fun = self.__interpolator__, **kwargs, : _fun(_dims, _obj, *args, **kwargs)

    def __setitem__(self, key, value):
        if value is None:
            value = np.zeros(shape=self.dimensions.shape)

        if isinstance(value, collections.abc.Mapping):
            self.__data__.setdefault(key, self.copy()).__update__(value)
        else:
            super().__setitem__(key, value)

    #     return res
    # if isinstance(dims, np.ndarray):
    #     if len(dims.shape) == 1:
    #         self._grid = dims
    #         self._grid_shape = dims.shape[0]
    #         self._axis_rank = 1
    #     else:
    #         self._grid = [x for x in dims]
    #         self._grid_shape = dims.shape[1:]
    #         self._axis_rank = dims.shape[0]
    # elif not isinstance(dims, list):
    #     raise TypeError(f"Illegal dimensions type {type(dims)}! ( only support list or numpy.ndarray)")
    # elif all([type(p) is int for p in dims]):
    #     self._grid = np.array(dims)
    #     self._grid_shape = len(dims)
    #     self._axis_rank = 1
    # elif all([isinstance(p, np.ndarray) for p in dims]):
    #     self._grid = dims
    #     self._grid_shape = dims[0].shape
    #     self._axis_rank = len(dims)

    # @property
    # def __dimensions__(self):
    #     return self.__dimensions__

    # def __missing__(self, key):
    #     super().__setitem__(key, np.zeros(shape=self._dimensions))
    #     # return super().setdefault(key, np.zeros(shape=self._dimensions))
    #     return super().__getitem__(key)


# class EqProfiles2DFreeGS(Profiles):

#     def __init__(self, backend, dims=None, *args, **kwargs):
#         super().__init__(dims or [129, 129], **kwargs)
#         self._backend = backend
#         pprint.pprint(self.__dict__)


if __name__ == "__main__":
    a = Profiles(np.linspace(0, 1, 10))
    a.b = {}
    a.b.c = 5

    # _, d = a.b.f.__push_back__()
    # d.text = "hellow world"
    pprint.pprint(a)
    pprint.pprint(a.b.d.shape)
    pprint.pprint(a.dimensions)

    # pprint.pprint(a.b.c.dimensions)
    # a.dimensions.fill(0)
    # pprint.pprint(a.dimensions)

    # pprint.pprint(a.b.c.dimensions)
    # pprint.pprint(a.b.e.__dimensions__)
