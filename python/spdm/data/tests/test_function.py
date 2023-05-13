import unittest

import numpy as np
from scipy import constants
from spdm.utils.logger import logger
from spdm.data.Function import Expression, Function, PiecewiseFunction


class TestFunction(unittest.TestCase):

    def test_type(self):
        x = np.linspace(0, 1.0, 128)
        y = np.sin(x*constants.pi*2.0)
        fun = Function[int](y, x)
        self.assertEqual(fun.__type_hint__, int)

    def test_spl(self):
        x = np.linspace(0, 1.0, 128)
        y = np.sin(x*constants.pi*2.0)

        fun = Function(y, x)

        x2 = np.linspace(0, 1.0, 64)
        y2 = np.sin(x2*constants.pi*2.0)

        self.assertLess(np.mean((y2-fun(x2))**2), 1.0e-16)  # type: ignore

    def test_expression(self):
        x = np.linspace(0, 1.0, 128)
        y = np.sin(x*constants.pi*2.0)
        fun = Function(y, x)

        expr = fun*2.0

        self.assertTrue(type(expr) is Expression)

    def test_operator(self):
        x = np.linspace(0, 1, 128)
        y = np.linspace(0, 2, 128)
        fun = Function(y, x)

        expr = fun == y

        logger.debug(expr)
        logger.debug(np.all(expr))

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
        fun = Function(y*2, x)

        self.assertTrue(np.all(fun == y * 2))

    def test_np_fun(self):
        x = np.linspace(0, 1, 128)
        y = np.linspace(0, 2, 128)
        fun = Function(y, x)

        self.assertTrue(type(fun+1) is Expression)
        self.assertTrue(type(fun*2) is Expression)
        self.assertTrue(type(np.sin(fun)) is Expression)

    # def test_different_x_domain(self):
    #     x0 = np.linspace(0, 2, 21)
    #     x1 = np.linspace(1, 3, 21)
    #     x2 = np.linspace(1, 2, 11)

    #     y0 = Function(lambda x: x, x0)
    #     y1 = Function(lambda x: x*2, x1)
    #     y2 = y0 + y1

    #     self.assertEqual(y2._mesh.min, 1)
    #     self.assertEqual(y2._mesh.max, 2)

    #     self.assertTrue(np.all(y2._mesh == x2))

    def test_picewise_function(self):
        r_ped = 0.9001  # np.sqrt(0.88)
        Cped = 0.2
        Ccore = 0.4
        chi = PiecewiseFunction([lambda x:x, lambda x: Cped],
                                [lambda x:x <= r_ped, lambda x:x > r_ped])
        x = np.linspace(0, 1, 101)
        logger.debug((chi*2)(x))


if __name__ == '__main__':
    unittest.main()
