import collections
import numpy as np
import pprint

from .Unit import Unit
from .Mesh import Mesh


class Coordinates:

    def __init__(self, *args, name=None, unit=None,  **kwargs) -> None:
        self._name = name.split(",") if isinstance(name, str) else name
        self._unit = [*map(Unit, unit.split(","))] if isinstance(unit, str) else unit
        self._mesh = Mesh(*args,  **kwargs)

    def __repr__(self) -> str:
        return f"""<{self.__class__.__name__} name=\"{self._name}\" />"""

    @property
    def name(self):
        return self._name

    @property
    def unit(self):
        return self._unit

    @property
    def mesh(self):
        return self._mesh

    def serialize(self):
        return {}

    @staticmethod
    def deserialize(cls, d):
        return Coordinates(d)
