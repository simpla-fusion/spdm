import unittest

import numpy as np
from scipy import constants
from spdm.data.Expression import _0 as _x
from spdm.data.Expression import _1 as _y
from spdm.data.Expression import _2 as _z
from spdm.data.Function import Expression, Function, Piecewise
from spdm.utils.logger import logger
from spdm.data.sp_property import SpPropertyClass, sp_property
from spdm.mesh.Mesh import Mesh
from spdm.data.Profile import Profile


class Doo(SpPropertyClass):

    @sp_property
    def grid(self) -> Mesh:
        return Mesh()

    x: np.ndarray = sp_property()
    a: Profile[int] = sp_property(coordinate1="../x")


class TestProfile(unittest.TestCase):

    def test_get(self):
        cache = {
            "a": np.random.rand(128),
            "x": np.linspace(0, 1, 128)
        }
        doo = Doo(cache)

        self.assertTrue(doo.a.__array__() is cache["a"])
        self.assertTrue(doo.a.domain[0] is cache["x"])


if __name__ == '__main__':
    unittest.main()
