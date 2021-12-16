import logging
import pprint
import sys
import unittest

import numpy as np
from scipy import constants
from spdm.common.logger import logger
from spdm.data.Function import Expression, Function, PiecewiseFunction


class TestFunction(unittest.TestCase):
    def test_spl(self):
        x = np.linspace(0, 1.0, 128)
        y = np.sin(x*constants.pi*2.0)

        fun = Function(x, y)

        x2 = np.linspace(0, 1.0, 64)
        y2 = np.sin(x2*constants.pi*2.0)

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

    def test_construct_from_expression(self):
        x = np.linspace(0, 1, 128)
        y = np.linspace(0, 2, 128)
        fun = Function(x, y*2)

        self.assertTrue(np.all(fun == y * 2))

    def test_np_fun(self):
        x = np.linspace(0, 1, 128)
        y = np.linspace(0, 2, 128)
        fun = Function(x, y)

        self.assertTrue(type(fun+1) is Expression)
        self.assertTrue(type(fun*2) is Expression)
        self.assertTrue(type(np.sin(fun)) is Expression)

    def test_different_x_domain(self):
        x0 = np.linspace(0, 2, 21)
        x1 = np.linspace(1, 3, 21)
        x2 = np.linspace(1, 2, 11)

        y0 = Function(x0, lambda x: x)
        y1 = Function(x1, lambda x: x*2)
        y2 = y0+y1

        self.assertEqual(y2.x_min, 1)
        self.assertEqual(y2.x_max, 2)

        self.assertTrue(np.all(y2.x_axis == x2))

    def test_picewise_function(self):
        r_ped = 0.9001  # np.sqrt(0.88)
        Cped = 0.2
        Ccore = 0.4
        chi = PiecewiseFunction([0, r_ped, 1.0],  [lambda x:x, lambda x: Cped])
        x = np.linspace(0, 1, 101)
        logger.debug((chi*2)(x))


if __name__ == '__main__':
    unittest.main()
