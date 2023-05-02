import unittest

import numpy as np
from scipy import constants
from spdm.utils.logger import logger
from spdm.grid.Grid import Grid


class TestGrid(unittest.TestCase):

    def test_nullgrid(self):
        grid = Grid()
        self.assertEqual(grid.type, None)
        self.assertEqual(grid.units, ["-"])
        self.assertEqual(grid.geometry, None)

    def test_structured_grid(self):
        from spdm.grid.StructuredMesh import StructuredMesh

        self.assertRaisesRegexp(
            TypeError, "Can't instantiate abstract class StructuredMesh with abstract method geometry", StructuredMesh, [10, 10])

    def test_uniform_mesh(self):
        grid = Grid("uniform")
        self.assertEqual(grid.type, "uniform")
        self.assertEqual(grid.units, ["-"])
        self.assertEqual(grid.geometry, None)


if __name__ == '__main__':
    unittest.main()
