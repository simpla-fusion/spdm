import unittest

import numpy as np
from scipy import constants
from spdm.data.Expression import Expression, Piecewise, Variable
from spdm.utils.logger import logger

TWOPI = constants.pi*2.0


class TestExpression(unittest.TestCase):

    def test_variable(self):
        _x = Variable(0, "x")
        _y = Variable(1, "y")

        self.assertEqual(_x.__name__, "x")
        self.assertEqual(str(_x+_y), "x + y")

    def test_expression(self):
        _x = Variable(0, "x")
        _y = Variable(1, "y")
        x = np.linspace(0.0, 1.0, 10)
        y = np.linspace(1.0, 2.0, 10)
        expr = np.sqrt(_x/_y)
        self.assertEqual(expr(1, 2), np.sqrt(0.5))
        self.assertTrue(np.allclose(expr(x, y), np.sqrt(x/y)))

    def test_picewise(self):
        _x = Variable(0)

        r_ped = 0.90  # np.sqrt(0.88)
        Cped = 0.2
        Ccore = 0.4
        chi = Piecewise([_x*2*Ccore, Cped], [_x < r_ped, _x >= r_ped])
        self.assertEqual(chi(0.5), (0.5*2*Ccore))
        self.assertEqual(chi(0.95), Cped)

        x = np.linspace(0, 1, 101)

        res = (chi**2)(x)

        self.assertTrue(np.allclose(res[x < r_ped], (x[x < r_ped]*2*Ccore)**2))
        self.assertTrue(np.allclose(res[x >= r_ped], (Cped)**2))


if __name__ == '__main__':
    unittest.main()
