import pprint
import sys
import unittest

from spdm.numlib import np,scipy

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

    def test_ufunc_expression(self):

        def func(x): return x**2

        x = np.linspace(0, 1.0, npoints)
        y = np.vectorize(func)(x)*3.1456
        p = Profile(func, axis=x)*3.1456

        self.assertEqual(type(p), ProfileExpression)
        self.assertTrue(all(p[:] == y[:]))

    def test_expression(self):

        x = np.linspace(0, 1.0, npoints)
        y, z = np.random.rand(2, npoints)

        r = y**2 + z/2.0

        Y = Profile(y, axis=x, description={"name": "Y"})
        Z = Profile(z, axis=x, description={"name": "Z"})

        R = Y**2 + Z/2.0
        self.assertEqual(type(R), Profile)
        self.assertTrue(all(R[:] == r[:]))

        R1 = np.float64(-2.0) * (Y + 1)

        self.assertEqual(type(R1), Profile)

        self.assertTrue(all(R1[:] == -2.0*(y+1.0)))

    def test_profile_integral(self):

        x = np.linspace(0, 10, npoints)
        y = np.sin(x)  # np.random.rand(npoints)
        z = scipy.integrate.cumtrapz(y, x, initial=0.0)

        Y = Profile(np.sin, axis=x)
        Z = Y.integral

        logger.debug((Z[:]==z[:]))
        self.assertTrue(all(Y[:] == y[:]))
        self.assertTrue(all(Z[:] == z[:]))

        Y1 = Y*2
        Z1 = Y1.integral
        z1 = scipy.integrate.cumtrapz(y*2, x, initial=0.0)
        self.assertTrue(all(Z1.value == z1[:]))

    def test_profiles_setitem(self):

        def func(x): return x**2

        x = np.linspace(0, 1.0, npoints)

        ps = Profiles(axis=x)
        ps["a"] = func
        y = func(x)

        self.assertTrue(all(ps.a[:] == y[:]))

    def test_profiles_setitem_expression(self):

        def func(x): return x**2

        x = np.linspace(0, 10, npoints)

        ps = Profiles(axis=x)
        y = func(x)
        r = np.sqrt(y+1.2345)

        Y = Profile(func, axis=x)

        ps["a"] = np.sqrt(Y+1.2345)

        self.assertTrue(type(Y + 1.2345), ProfileExpression)
        self.assertTrue(all(ps.a[:] == r[:]))


if __name__ == '__main__':
    from spdm.data.Profile import Profile, ProfileFunction, ProfileExpression, Profiles
    from spdm.util.logger import logger

    unittest.main()
