import tempfile
import unittest

import numpy as np
from spdm.utils.logger import logger
from spdm.data.File import File


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

    def test_xml(self):
        device_desc = File("/home/salmon/workspace/fytok_data/mapping/ITER/imas/3/static/config.xml",
                           format="XML").read()
        # logger.debug(device_desc.get({"wall", "pf_active", "tf", "magnetics"}).dump())
        # {"wall", "pf_active", "tf", "magnetics"}
        logger.debug(device_desc.child("wall/description_2d/limiter").dump())


if __name__ == '__main__':
    unittest.main()
