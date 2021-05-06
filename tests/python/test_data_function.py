import logging
import pprint
import sys
import unittest
import scipy.constants
import numpy as np
from spdm.data.Function import Function
from spdm.util.logger import logger


class TestFunction(unittest.TestCase):
    def test_init(self):
        x = np.linspace(0, 1, 128)

        fun = Function(x, 1.0)

        logger.debug(fun)

    def test_spl(self):
        x = np.linspace(0, 1.0, 128)
        y = np.sin(x*scipy.constants.pi*2.0)

        fun = Function(x, y, is_periodic=True)

        x2 = np.linspace(0, 1.0, 64)
        y2 = np.sin(x2*scipy.constants.pi*2.0)

        self.assertLess(np.mean((y2-fun(x2))**2), 1.0e-16)

    def test_operator(self):
        x = np.linspace(0, 1, 128)
        y = np.linspace(0, 2, 128)
        fun = Function(x, y)

        self.assertTrue(np.all(-fun == -y))
        self.assertTrue(np.all(fun + 2 == y + 2))
        self.assertTrue(np.all(fun - 2 == y - 2))
        self.assertTrue(np.all(fun * 2 == y * 2))
        self.assertTrue(np.all(fun / 2 == y / 2))
        self.assertTrue(np.all(fun ** 2 == y ** 2))
        # self.assertTrue(np.all(fun @ fun == y)


if __name__ == '__main__':
    unittest.main()
