import collections
import numpy as np

import pprint


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
        if isinstance(coord, str):
            self._name = coord
            self._data = None
        else:
            self._name = None
            self._data = coord

    def __repr__(self) -> str:
        msg = ""
        if isinstance(self._data, collections.abc.Mapping):
            for k, d in self._data.items():
                msg += f"""\t<{k}> {pprint.pformat(d)} </{k}>\n"""
        else:
            msg=pprint.pformat(self._data)

        if not self._name:
            name=""
        else:
            name=f" name=\"{self._name}\""
        return f"""<{self.__class__.__name__}{name}>
        {msg}
        </{self.__class__.__name__}>
        """

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
