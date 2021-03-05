import collections
import unittest

import numpy as np
from spdm.data.Node import Node, _next_
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

        g.a.b = 5
        g.d.c.d[_next_] = "hello world!"
        g.d.c.e = "  world!"
        # logger.debug( g.d.c.g.h)
        g.b = [1, 2, 3, 4, {"c": {"d": 2345}}]
        g.c = Quantity(5, unit="g")
        # g.b[_next_] = g.d.c.g.h

        logger.debug(g)
        logger.debug(g.a.d.e)
        logger.debug(g.b[2])
        logger.debug(g.c.unit)
        logger.debug(g.c)
        logger.debug(len(g))
        logger.debug("b" in g)

        # for item in g:
        #     logger.debug(item)


if __name__ == '__main__':
    unittest.main()
