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

    psi: np.ndarray = sp_property()
    phi: Profile[int] = sp_property(coordinate1="../psi")


class TestProfile(unittest.TestCase):

    def test_get(self):
        cache = {
            "phi": np.random.rand(128),
            "psi": np.linspace(0, 1, 128)
        }
        doo = Doo(cache)
        logger.debug(doo.phi.name)

        doo.phi(np.linspace(0,1.0,256))

        self.assertTrue(np.all(np.isclose(doo.phi.__array__(), cache["phi"])))
        self.assertTrue(np.all(np.isclose(doo.phi.mesh, cache["psi"])))


if __name__ == '__main__':
    unittest.main()
