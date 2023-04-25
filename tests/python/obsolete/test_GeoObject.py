import unittest

from spdm.geometry.Point import Point
from spdm.geometry.Curve import Curve
from spdm.geometry.Surface import Surface
from spdm.geometry.CubicSplineCurve import CubicSplineCurve
import numpy as np, constants
from spdm.utils.logger import logger


class TestGeoObject(unittest.TestCase):
    def test_point(self):

        p1 = Point(1.0)
        self.assertEqual(p1.rank, 0)
        self.assertEqual(p1.ndims, 1)

        p2 = Point(1.0, 2.0)

        self.assertEqual(p2.rank, 0)
        self.assertEqual(p2.ndims, 2)

        d = [1.0, 2.0, 3.0]
        p3 = Point(d)

        self.assertEqual(p3.rank, 0)
        self.assertEqual(p3.ndims, 3)
        self.assertTrue(p3.is_closed)
        u = np.linspace(0, 1, 11)
        self.assertTrue(np.allclose(p3.dl(u), 0*u))

        self.assertTrue(np.allclose(p3.bbox, [[d[i], d[i]] for i in range(p3.ndims)]))

    def test_curve(self):
        nu = 17
        xy = np.random.random([nu, 2])
        xy[-1] = xy[0]
        c2 = Curve(xy)
        self.assertEqual(c2.rank, 1)
        self.assertEqual(c2.ndims, 2)
        self.assertEqual(c2.shape, [nu])
        self.assertTrue(np.allclose(c2.points(), xy))
        self.assertTrue(c2.is_closed)
        self.assertTrue(np.allclose(c2.bbox, [[xy[:, i].min(), xy[:, i].max()] for i in range(c2.ndims)]))

        xyz = np.random.random([nu, 3])
        c3 = Curve(xyz)
        xyz[-1] = xyz[0]+1.0

        self.assertEqual(c3.rank, 1)
        self.assertEqual(c3.ndims, 3)
        self.assertEqual(c3.shape, [nu])
        self.assertTrue(np.allclose(c3.points(), xyz))
        self.assertFalse(c3.is_closed)
        self.assertTrue(np.allclose(c3.bbox, [[xyz[:, i].min(), xyz[:, i].max()] for i in range(c3.ndims)]))

    def test_surface(self):
        nu = 10
        nv = 7
        xv, yv = np.meshgrid(np.linspace(0, 5, nu), np.linspace(-3, 2, nv),  indexing='ij')

        points2 = np.moveaxis(np.asarray([xv, yv]), 0, -1)

        s2 = Surface(points2)
        self.assertEqual(s2.rank, 2)
        self.assertEqual(s2.ndims, 2)
        self.assertEqual(s2.shape, [nu, nv])

        zw = np.random.random([nu, nv])
        xyz = np.asarray([xv, yv, zw])
        points3 = np.moveaxis(xyz, 0, -1)

        s3 = Surface(points3)
        self.assertEqual(s3.rank, 2)
        self.assertEqual(s3.ndims, 3)
        self.assertEqual(s3.shape, [nu, nv])
        self.assertTrue(np.allclose(s3.bbox, [[xyz[i].min(), xyz[i].max()] for i in range(s3.ndims)]))

    def test_spline_curve2d(self):
        nu = 17
        u0 = np.linspace(0, 1.0, nu)
        u0 = np.flip(np.roll(u0, 4))

        xy = np.asarray([np.sin((u0)*constants.pi*2), np.cos((u0)*constants.pi*2)])

        pts = np.moveaxis(xy, 0, -1)

        curv = CubicSplineCurve(pts, [u0])

        self.assertTrue(curv.is_closed)

        u1 = np.linspace(0, 1.0, nu*2)

        pts1 = curv.points(u1)

        self.assertTrue(np.allclose(pts1[:, 0], np.sin((u1)*constants.pi*2), rtol=0.5/nu))

        self.assertTrue(np.allclose(pts1[:, 1], np.cos((u1)*constants.pi*2), rtol=0.5/nu))

        self.assertTrue(np.allclose(curv.dl(u1), constants.pi*2/(2*nu), atol=0.5/nu))


if __name__ == '__main__':
    unittest.main()
