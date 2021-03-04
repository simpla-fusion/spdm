import numpy as np


class Coordinates:

    # @staticmethod
    # def __new__(cls, *args, **kwargs):

    #     return object.__new__(cls)

    def __init__(self, coord, *args, **kwargs) -> None:
        self._ndmis = 1
        self._ndims_manifold = 1
        self._defined_domain = 1
        self._dataset_shape = []
        self._mesh = np.ndarray(self._dataset_shape)
        self._name = coord

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name=\"{self._name}\" />"

    @property
    def __name__(self):
        return self._name

    def serialize(self):
        return {}

    @staticmethod
    def deserialize(cls, d):
        return Coordinates(d)

    @property
    def dataset_shape(self):
        return self._dataset_shape
