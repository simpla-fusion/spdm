import unittest

import numpy as np
from scipy import constants
from spdm.data.Expression import _0 as _x
from spdm.data.Expression import _1 as _y
from spdm.data.Expression import _2 as _z
from spdm.data.Field import Field
from spdm.utils.logger import logger


class TestField(unittest.TestCase):

    def test_spl2d(self):

        x = np.linspace(0, 1, 128)
        y = np.linspace(0, 2, 128)
        g_x, g_y = np.meshgrid(x, y)

        z = np.sin(g_x*constants.pi*2.0)*np.cos(g_y*constants.pi*2.0)

        fun = Field(np.sin(_x*constants.pi*2.0)*np.cos(_y*constants.pi*2.0), x, y, cycles=[1, 1])

        z2 = fun(g_x, g_y)

        self.assertTrue(np.all(np.isclose(z, z2)))


if __name__ == '__main__':
    unittest.main()
