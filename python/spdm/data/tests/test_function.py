import unittest

import numpy as np
from scipy import constants
from spdm.data.Expression import Variable
from spdm.data.Function import Expression, Function, Piecewise
from spdm.utils.logger import logger

TWOPI = constants.pi*2.0


class TestFunction(unittest.TestCase):

    def test_type(self):
        x = np.linspace(0, 1.0, 128)
        y = np.sin(x*TWOPI)
        fun = Function[int](y, x)
        self.assertEqual(fun.__type_hint__, int)

    def test_spl(self):
        x = np.linspace(0, 1.0, 128)
        y = np.sin(x*TWOPI)

        fun = Function(y, x)

        x2 = np.linspace(0, 1.0, 64)
        y2 = np.sin(x2*TWOPI)

        self.assertLess(np.mean((y2-fun(x2))**2), 1.0e-16)  # type: ignore

    def test_expression(self):
        x = np.linspace(0, 1.0, 128)
        y = np.sin(x*TWOPI)
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

    def test_picewise(self):
        _x = Variable(0, "x")

        r_ped = 0.90  # np.sqrt(0.88)
        Cped = 0.2
        Ccore = 0.4
        chi = Piecewise([_x*2*Ccore, Cped], [_x < r_ped, _x >= r_ped])
        self.assertEqual(chi(0.5), (0.5*2*Ccore))
        self.assertEqual(chi(0.95), Cped)

        x = np.linspace(0, 1, 101)

        res = (chi**2)(x)

        self.assertTrue(np.all(np.isclose(res[x < r_ped], (x[x < r_ped]*2*Ccore)**2)))
        self.assertTrue(np.all(np.isclose(res[x >= r_ped], (Cped)**2)))

    def test_dydx(self):

        x = np.linspace(0, TWOPI, 512)

        _x = Variable(0, "x")

        Y = Function(np.sin(_x), x, periods=[1])

        self.assertTrue(np.allclose(np.sin(x), Y(x), rtol=1.0e-4))

        self.assertTrue(np.allclose(np.cos(x), Y.d()(x), rtol=1.0e-4))
        # logger.debug((-(TWOPI**2)*np.sin(x))[:10])
        # logger.debug(Y.d(2)(x)[:10])
        # self.assertTrue(np.allclose(-(TWOPI**2)*np.sin(x), Y.d(2)(x), rtol=0.10))

    def test_spl2d(self):

        x = np.linspace(0, TWOPI, 128)
        y = np.linspace(0, 2*TWOPI, 128)
        g_x, g_y = np.meshgrid(x, y)

        z = np.sin(g_x)*np.cos(g_y)

        _x = Variable(0, "x")

        _y = Variable(1, "y")

        fun = Function(np.sin(_x)*np.cos(_y), x, y, periods=[1, 1])

        z2 = fun(g_x, g_y)

        self.assertTrue(np.allclose(z, z2, rtol=1.0e-4))

    def test_pd2(self):

        x = np.linspace(0, TWOPI, 128)
        y = np.linspace(0, 2*TWOPI, 128)

        g_x, g_y = np.meshgrid(x, y)

        _x = Variable(0, "x")
        _y = Variable(1, "y")

        Z = Function(np.sin(_x)*np.cos(_y), x, y, periods=[1, 1])

        self.assertTrue(np.allclose(np.sin(g_x)*np.cos(g_y),  Z(g_x, g_y), rtol=1.0e-4))

        self.assertTrue(np.allclose(np.cos(g_x)*np.cos(g_y),  Z.pd(1, 0)(g_x, g_y), rtol=1.0e-4))
        
        self.assertTrue(np.allclose(- np.sin(g_x)*np.sin(g_y),  Z.pd(0, 1)(g_x, g_y), rtol=1.0))

        # self.assertTrue(np.allclose(-TWOPI*np.cos(g_x) *
        #                 np.sin(g_y),  Z.pd(1, 1)(g_x, g_y), rtol=1.0e-4))

        # dzdy = -TWOPI*np.sin(g_x)*np.sin(g_y)
        # dZdy = Z.pd(0, 1)(g_x, g_y)
        # logger.debug(np.count_nonzero(~np.isclose(dzdy, dZdy, rtol=1.0e-4)))
        # self.assertTrue(np.allclose(dzdy, dZdy, rtol=1.0e-4))

        # dZdxdy = Z.pd(1, 1)(g_x, g_y)
        # dzdxdy = -(TWOPI)*(TWOPI)*np.cos(g_x)*np.sin(g_y)
        # logger.debug(np.count_nonzero(~np.isclose(dzdxdy, dZdxdy, rtol=1.0e-4)))
        # self.assertTrue(np.allclose(dzdxdy, dZdxdy, rtol=1.0e-4))


if __name__ == '__main__':
    unittest.main()
