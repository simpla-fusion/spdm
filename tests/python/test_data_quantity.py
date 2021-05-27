import collections
import unittest

from spdm.numlib import np
from spdm.data.Node import Node, _next_
from spdm.data.Quantity import Quantity
from spdm.data.Coordinates import Coordinates
from spdm.util.logger import logger
from spdm.data.Unit import Unit
import pprint

dobj = Quantity


class TestQuantity(unittest.TestCase):

    def test_create(self):
        x0 = Quantity(np.linspace(1, 1.0, 10), unit="m")
        s0 = dobj(1.0, dtype=float, coordinates=x0, unit="s")
        d0 = dobj(1, dtype=float, coordinates=x0, unit="m")
        d1 = dobj(2, dtype=float, coordinates=x0, unit="kg")
        d2 = dobj(3.0, dtype=float, coordinates=x0, unit="newton")

        res = d0*d1/(s0**2)-d2*2
        # res = (d1-d1)*d1/(s0**(2)) + d2*2

        # logger.debug(d0.unit)
        # logger.debug(d1.unit)
        # logger.debug(res[:, 1].unit.is_dimensionless)
        # logger.debug(res.unit)
        logger.debug((res))
        logger.debug(res.unit)
        logger.debug(type(res))

    def test_quantity_with_coordinates(self):
        NX = 128
        NY = 128

        dim0 = np.linspace(0, 1, NX)
        dim1 = np.linspace(0, 1, NY)

        coord = Coordinates(dim0, dim1, name="x,y", unit="m")

        logger.debug(coord.name)
        logger.debug(coord.unit)
        logger.debug(coord.mesh.mesh)

    #     logger.debug(x0)
    #     logger.debug((y0.unit))
    #     logger.debug((y0.coordinates))
    # #     g = dobj()

    #     g.a.b = 5
    #     g.d.c.d[_next_] = "hello world!"
    #     g.d.c.e = "  world!"
    #     # logger.debug( g.d.c.g.h)
    #     g.b = [1, 2, 3, 4, {"c": {"d": 2345}}]
    #     g.c = 5
    #     # g.c = Quantity(5, unit="g")
    #     # g.b[_next_] = g.d.c.g.h

    #     logger.debug(g)
    #     # logger.debug(g.a.d.e)
    #     logger.debug(g.b[2])
    #     # logger.debug(g.c.unit)
    #     logger.debug(g.c)
    #     logger.debug(len(g))
    #     logger.debug("b" in g)

        # for item in g:
        #     logger.debug(item)


if __name__ == '__main__':
    unittest.main()
