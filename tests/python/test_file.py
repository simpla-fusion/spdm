import tempfile
import unittest

import h5py
import numpy as np
from spdm.common.logger import logger
from spdm.data.File import File
from spdm.data.Path import Path


class TestFile(unittest.TestCase):
    data = {
        "a": [
            "hello world {name}!",
            "hello world2 {name}!",
        ],
        "b": [1.0, 2, 3, 4],
        "c": "I'm {age}!",
        "d": {
            "e": "{name} is {age}",
            "f": "{address}",
            "g": {"a": 1, "b": 2}
        },
        "h": np.random.random([10, 10])
    }

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="spdm_")
        return super().setUp()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        self.temp_dir = None
        return super().tearDown()

    def test_h5(self):
        f_name = "/home/salmon/workspace/output/test.h5"
        # f_name = f"{self.temp_dir.name}/test.h5"

        with File(f_name, mode="w") as f_out:
            f_out.write(self.data)

        with File(f_name, mode="r") as f_in:
            res = f_in.read()

        self.assertListEqual(list(res.get("a")), self.data["a"])
        self.assertListEqual(list(res.get("b")), self.data["b"])
        self.assertEqual(res.get("d.e"), self.data["d"]["e"])

        self.assertDictEqual(res.get("d"), self.data["d"])

        self.assertTrue(np.array_equal(res.get("h"), self.data["h"]))

    def test_xml(self):
        device_desc = File("/home/salmon/workspace/fytok/data/mapping/ITER/imas/3/static/config.xml",
                           format="XML").read()
        # logger.debug(device_desc.get({"wall", "pf_active", "tf", "magnetics"}).dump())
        # {"wall", "pf_active", "tf", "magnetics"}
        logger.debug(device_desc.child("wall", 'description_2d', 'limiter').pull().dump())


if __name__ == '__main__':
    unittest.main()
