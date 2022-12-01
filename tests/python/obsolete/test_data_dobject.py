import importlib
import pprint
import sys
import unittest
import numpy as np

from spdm.util.logger import logger
from spdm.data.Quantity import Quantity
dobj = Quantity


class TestAttribute(unittest.TestCase):

    def test_create(self):
        s = dobj(np.ones((3, 3)), dtype=int, unit="s")
        d0 = dobj(np.ones((3, 3)), dtype=int, unit="m")
        d1 = dobj(np.ones((3, 3)), dtype=int, unit="kg")
        d2 = dobj(np.ones((3, 3)), dtype=int, unit="newton")

        res = d0*d1/(s**(2)) + d2*2

        # logger.debug(d0.unit)
        # logger.debug(d1.unit)
        # # logger.debug(res[:, 1].unit.is_dimensionless)
        # logger.debug(res.unit)
        logger.debug(res)
        logger.debug(type(res))


if __name__ == '__main__':
    unittest.main()
