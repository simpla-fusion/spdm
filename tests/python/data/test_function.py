import unittest

import numpy as np
from scipy import constants
from spdm.data.Expression import Expression, Variable
from spdm.data.Function import Function
from spdm.utils.logger import logger

TWOPI = constants.pi*2.0


class TestFunction(unittest.TestCase):

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

        self.assertTrue(np.allclose(fun, y * 2))

    def test_np_fun(self):
        x = np.linspace(0, 1, 128)
        y = np.linspace(0, 2, 128)

        fun = Function(y, x)

        self.assertTrue(type(fun+1) is Expression)
        self.assertTrue(type(fun*2) is Expression)
        self.assertTrue(type(np.sin(fun)) is Expression)

    def test_constant_fun(self):
        value = 1.2345
        xmin = 0.0
        xmax = 1.0
        ymin = 2.0
        ymax = 3.0

        dims_a = np.linspace(xmin, xmax, 5),  np.linspace(ymin, ymax, 5)

        fun1 = Function(value)

        xa, ya = np.meshgrid(*dims_a)

        self.assertTrue(np.allclose(fun1(xa, ya), value))

        fun2 = Function(value, *dims_a)

        dims_b = np.linspace(xmin-0.5, xmax+0.5, 10), np.linspace(ymin-0.5, ymax+0.5, 10)

        xb, yb = np.meshgrid(*dims_b)
        marker = (xb >= xmin) & (xb <= xmax) & (yb >= ymin) & (yb <= ymax)

        e_a = np.full_like(xa, value, dtype=float)

        e_b = np.full_like(xb, value, dtype=float)

        e_b[~marker] = np.nan

        self.assertTrue(np.allclose(fun2(xa, ya), e_a))

        self.assertTrue(np.allclose(fun2(xb, yb), e_b, equal_nan=True))

    def test_spl(self):
        x = np.linspace(0, 1.0, 128)
        y = np.sin(x*TWOPI)

        fun = Function(y, x)

        x2 = np.linspace(0, 1.0, 64)

        y2 = np.sin(x2*TWOPI)

        self.assertTrue(np.allclose(y2, fun(x2)))

    def test_delta_fun(self):

        p = 0.5
        value = 1.2345

        fun0 = Function(value, p)

        self.assertTrue(np.isclose(fun0(p), value))

        fun1 = Function([value], [p])

        self.assertTrue(np.isclose(fun1(p), value))

        x = np.linspace(0, 1, 11)

        mark = np.isclose(x, p)

        # logger.debug(fun1(x))
        self.assertTrue(np.allclose(fun1(x)[mark], value))
        self.assertTrue(np.all(np.isnan(fun1(x)[~mark])))

    def test_delta_nd(self):

        p = [0.5, 0.4]
        value = 1.2345

        fun0 = Function(value, *p)

        self.assertTrue(np.isclose(fun0(*p), value))

        dimx = np.linspace(0, 1, 11)
        dimy = np.linspace(0, 1, 11)

        x, y = np.meshgrid(dimx, dimy)

        mark = np.isclose(x, p[0]) & np.isclose(y, p[1])

        self.assertTrue(np.allclose(fun0(x, y)[mark], value))
        self.assertTrue(np.all(np.isnan(fun0(x, y)[~mark])))


if __name__ == '__main__':
    unittest.main()
