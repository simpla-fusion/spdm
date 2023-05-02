import unittest

import numpy as np
from scipy import constants
from spdm.utils.logger import logger
from spdm.geometry.GeoObject import GeoObject


class TestGrid(unittest.TestCase):

    def test_line(self):
        p0 = (0, 0)
        p1 = (1, 1)
        gobj = GeoObject("line", p0, p1)
        self.assertEqual(type(gobj).__name__, "Line")
        from spdm.geometry.Line import Line
        self.assertIsInstance(gobj, Line)

    def test_line2(self):
        from spdm.geometry.Line import Line
        p0 = (0, 0, 0)
        p1 = (1, 2, 3)
        gobj = Line(p0, p1)

        self.assertEqual(type(gobj).__name__, "Line")
        self.assertEqual(gobj.p0[:], p0)
        self.assertEqual(gobj.p1[:], p1)

        logger.debug(gobj.points(np.linspace(0, 1.0, 10)))


if __name__ == '__main__':
    unittest.main()
