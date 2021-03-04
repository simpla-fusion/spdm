import collections
import unittest

import numpy as np
from spdm.data.Node import Node
from spdm.data.Quantity import Quantity
from spdm.util.logger import logger
from spdm.data.Coordinates import Coordinates


class QuantityTree(Node):

    def __init__(self, *args, coordinates=None, **kwargs) -> None:
        if coordinates is not None:
            coordinates = coordinates if isinstance(coordinates, Coordinates) else Coordinates(coordinates)

        if coordinates is not None:
            self._coordinates = coordinates

        super().__init__(*args, coordinates=coordinates, **kwargs)

    @property
    def coordinates(self):
        return getattr(self, "_coordinates", None) or getattr(self._value, "coordinates", None) or getattr(self._parent, "coordinates", None)

    def __update__(self, value, *args, coordinates=None, **kwargs):
        if isinstance(value, (collections.abc.Mapping, collections.abc.Sequence)) and not isinstance(value, str):
            super().__update__(value, *args, **kwargs)
        elif isinstance(value, Quantity):
            self._value = value
        elif value is not None:
            self._value = Quantity(value, *args, coordinates=coordinates or self.coordinates, **kwargs)


dobj = QuantityTree


class TestQuantity(unittest.TestCase):

    # def test_create(self):
    #     s = dobj(np.ones((3, 3)), dtype=int, unit="s")
    #     d0 = dobj(np.ones((3, 3)), dtype=int, unit="m")
    #     d1 = dobj(np.ones((3, 3)), dtype=int, unit="kg")
    #     d2 = dobj(np.ones((3, 3)), dtype=int, unit="newton")

    #     res = (2*d0)*d1/(s**(2)) + d2*2

    #     # logger.debug(d0.unit)
    #     # logger.debug(d1.unit)
    #     # logger.debug(res[:, 1].unit.is_dimensionless)
    #     # logger.debug(res.unit)
    #     logger.debug(res)
    #     logger.debug(res.unit)
    #     logger.debug(type(res))

    def test_quantity_group(self):

        g = dobj(coordinates="cartesian")

        g["a"] = {"b": 5}
        g["b"] = [1, 2, 3, 4, {"c": {"d": 2345}}]
        g["c"] = Quantity(5, unit="g")

        logger.debug(g)
        logger.debug(g["a"]["b"].coordinates)
        logger.debug(g["c"].unit)
        logger.debug(g["c"])


if __name__ == '__main__':
    unittest.main()
