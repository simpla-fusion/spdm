import pprint
import sys
import unittest

import numpy as np

sys.path.append("/home/salmon/workspace/SpDev/SpDB")

npoints = 129


class TestProfile(unittest.TestCase):

    def test_new_from_ndarray(self):
        x = np.linspace(0, 1.0, npoints)
        y = np.random.rand(npoints)
        p = y.view(Profile)
        p._axis = x
        Profile(y, axis=x)
        self.assertTrue(all(p[:] == y[:]))

    def test_new_from_ufunc(self):

        def func(x): return x**2

        x = np.linspace(0, 1.0, npoints)
        y = np.vectorize(func)(x)
        p = Profile(func, axis=x)

        self.assertTrue(all(p[:] == y[:]))


if __name__ == '__main__':
    from spdm.util.Profiles import Profile

    unittest.main()
