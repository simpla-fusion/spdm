import pprint
import sys
import unittest

import numpy as np
import scipy.integrate
sys.path.append("/home/salmon/workspace/SpDev/SpDB")

npoints = 129


class TestRefResolver(unittest.TestCase):

    def test_validate(self):
        resolver = RefResolver(
            base_uri="https://fusionyun.org/schema/draft-00",
            prefetch="pkgdata://spdm/schemas/",
            alias=[["*",   "/home/salmon/workspace/SpDev/SpDB/examples/data/FuYun/modules/*/fy_module.yaml", ],
                   ["*", "/fuyun/modules/*/fy_module.yaml"]])

        logger.debug(resolver.fetch("physics/genray/201213-gompi-2019b"))


if __name__ == '__main__':
    from spdm.util.RefResolver import RefResolver
    from spdm.logger import logger

    unittest.main()
