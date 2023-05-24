import unittest

import numpy as np
from scipy import constants
from spdm.data.Expression import _0 as _x
from spdm.data.Expression import _1 as _y

from spdm.data.Field import Field
from spdm.utils.logger import logger

TWOPI = constants.pi*2.0


class TestField(unittest.TestCase):

    def test_spl2d(self):

        x = np.linspace(0, 1, 128)
        y = np.linspace(0, 2, 128)
        g_x, g_y = np.meshgrid(x, y)

        z = np.sin(g_x*TWOPI)*np.cos(g_y*TWOPI)

        fun = Field(np.sin(_x*TWOPI)*np.cos(_y*TWOPI), x, y, mesh_periods=[1, 1], mesh_method="cubic")

        z2 = fun(g_x, g_y)

        self.assertTrue(np.allclose(z, z2, rtol=1.0e-4))

    def test_pd(self):

        x = np.linspace(0, TWOPI, 128)
        y = np.linspace(0, 2*TWOPI, 128)
        g_x, g_y = np.meshgrid(x, y)

        Z = Field(np.sin(_x)*np.cos(_y), x, y,   mesh_periods=[1, 1])

        self.assertTrue(np.allclose(np.sin(g_x)*np.cos(g_y),  Z(g_x, g_y), rtol=1.0e-4))
        
        self.assertTrue(np.allclose(np.cos(g_x)*np.cos(g_y),  Z.pd(1, 0)(g_x, g_y), rtol=1.0e-4))

        # dzdx = TWOPI*np.cos(g_x*TWOPI)*np.cos(g_y*TWOPI)
        # dZdx = Z.pd(1, 0)(g_x, g_y)
        # # logger.debug(np.count_nonzero(~np.isclose(dzdx, dZdx, rtol=1.0e-4)))
        # self.assertTrue(np.allclose(dzdx, dZdx, rtol=1.0e-4))

        # dzdy = -TWOPI*np.sin(g_x*TWOPI)*np.sin(g_y*TWOPI)
        # dZdy = Z.pd(0, 1)(g_x, g_y)
        # logger.debug(np.count_nonzero(~np.isclose(dzdy, dZdy, rtol=1.0e-4)))
        # self.assertTrue(np.allclose(dzdy, dZdy, rtol=1.0e-4))

        # dZdxdy = Z.pd(1, 1)(g_x, g_y)
        # dzdxdy = -(TWOPI)*(TWOPI)*np.cos(g_x*TWOPI)*np.sin(g_y*TWOPI)
        # logger.debug(np.count_nonzero(~np.isclose(dzdxdy, dZdxdy, rtol=1.0e-4)))
        # self.assertTrue(np.allclose(dzdxdy, dZdxdy, rtol=1.0e-4))


if __name__ == '__main__':
    unittest.main()
