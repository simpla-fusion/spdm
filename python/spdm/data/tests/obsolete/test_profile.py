import unittest

import numpy as np
from scipy import constants
from spdm.data.Expression import Variable
from spdm.data.Function import Function
from spdm.utils.logger import logger
from spdm.data.sp_property import SpTree, sp_property
from spdm.mesh.Mesh import Mesh
from spdm.data.Profile import Profile


class Doo(SpTree):

    @sp_property
    def grid(self) -> Mesh:
        return Mesh()

    psi: np.ndarray = sp_property()
    phi: Profile[float] = sp_property(coordinate1="../psi")


class TestProfile(unittest.TestCase):

    def test_get(self):
        cache = {
            "phi": np.random.rand(128),
            "psi": np.linspace(0, 1, 128)
        }
        doo = Doo(cache)

        doo.phi(np.linspace(0, 1.0, 256))

        self.assertTrue(np.all(np.isclose(doo.phi.__array__(), cache["phi"])))
        self.assertTrue(np.all(np.isclose(doo.phi.points[0], cache["psi"])))

    def test_prop_expr(self):
        cache = {
            "phi": Variable(0, "psi")*2,
            "psi": np.linspace(0, 1, 128)
        }

        doo = Doo(cache)

        doo.phi(np.linspace(0, 1.0, 256))

        self.assertTrue(np.allclose(doo.phi.__array__(), cache["psi"]*2.0))
        self.assertTrue(np.allclose(doo.phi.points[0], cache["psi"]))


if __name__ == '__main__':
    unittest.main()
