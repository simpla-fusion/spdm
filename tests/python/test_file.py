import unittest
import numpy as np
from spdm.data.Node import Node, Dict, List, _next_, _not_found_
from spdm.common.logger import logger
from copy import copy, deepcopy
from spdm.data.File import File
import tempfile
import h5py


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
        logger.debug(res.get("d"))
        self.assertDictEqual(res.get("d"), self.data["d"])

        self.assertTrue(np.array_equal(res.get("h"), self.data["h"]))


if __name__ == '__main__':
    unittest.main()
