import collections
import unittest

import numpy as np
from spdm.data.Node import Node, _next_
from spdm.data.Quantity import Quantity
from spdm.util.logger import logger
from spdm.data.Unit import Unit
import pprint

dobj = Quantity


class TestQuantity(unittest.TestCase):

    def test_create(self):
        s = dobj(np.ones((3, 3)), dtype=int, unit="s")
        d0 = dobj(np.ones((3, 3)), dtype=int, unit="m")
        d1 = dobj(np.ones((3, 3)), dtype=int, unit="kg")
        d2 = dobj(np.ones((3, 3)), dtype=int, unit="newton")

        res = (2*d0)*d1/(s**(2)) + d2*2

        # logger.debug(d0.unit)
        # logger.debug(d1.unit)
        # logger.debug(res[:, 1].unit.is_dimensionless)
        # logger.debug(res.unit)
        logger.debug(res)
        logger.debug(res.unit)
        logger.debug(type(res))

    # def test_quantity_group(self):

    #     g = dobj()

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
