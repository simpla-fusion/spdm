import unittest

import numpy as np
from scipy import constants
from spdm.utils.logger import logger
from spdm.data.Function import Expression, Function, PiecewiseFunction
from spdm.data.Function import _0 as _x
from spdm.data.Function import _1 as _y
from spdm.data.Function import _2 as _z


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
        r_ped = 0.90  # np.sqrt(0.88)
        Cped = 0.2
        Ccore = 0.4
        chi = PiecewiseFunction([_x*2*Ccore, Cped], [_x < r_ped, _x >= r_ped])
        self.assertEqual(chi(0.5), (0.5*2*Ccore))
        self.assertEqual(chi(0.95), Cped)

        x = np.linspace(0, 1, 101)

        res = (chi**2)(x)

        self.assertTrue(np.all(np.isclose(res[x < r_ped], (x[x < r_ped]*2*Ccore)**2)))
        self.assertTrue(np.all(np.isclose(res[x >= r_ped], (Cped)**2)))

    def test_spl2d(self):

        x = np.linspace(0, 1, 128)
        y = np.linspace(0, 2, 128)
        g_x, g_y = np.meshgrid(x, y)

        z = np.sin(g_x*constants.pi*2.0)*np.cos(g_y*constants.pi*2.0)

        fun = Function(np.sin(_x*constants.pi*2.0)*np.cos(_y*constants.pi*2.0), x, y, cycles=[1, 1])
        z2 = fun(g_x, g_y)

        self.assertTrue(np.all(np.isclose(z, z2)))


if __name__ == '__main__':
    unittest.main()
