import pprint
import sys
import unittest

from spdm.util.numlib import np
import scipy.integrate
sys.path.append("/home/salmon/workspace/SpDev/SpDB")

npoints = 129


class TestSpObject(unittest.TestCase):

    def test_primary_type(self):
        i = SpObject(1234, schema="integer")
        self.assertTrue(isinstance(i, int))
        self.assertEqual(i, 1234)

        f = SpObject(1234.0, schema="float")
        self.assertTrue(isinstance(f, float))
        self.assertEqual(f, 1234.0)

        s = SpObject(1234.0, schema="string")
        self.assertTrue(isinstance(s, str))
        self.assertEqual(s, "1234.0")

    def test_dataobject(self):
        f = File(schema={"$id": "file.netcdf", "path": "temp.nc"})
        self.assertTrue(isinstance(f, File))


if __name__ == '__main__':
    from spdm.util.SpObject import SpObject
    from spdm.data.File import File

    from spdm.util.logger import logger

    unittest.main()
