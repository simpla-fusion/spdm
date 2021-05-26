import importlib
import pprint
import sys
from spdm.util.numlib import np
import unittest
from spdm.util.logger import logger
from spdm.data.Profiles import Profiles
from spdm.data.Node import _next_


class TestProfiles(unittest.TestCase):
    def test_profile_initialize(self):
        axis = np.linspace(0, 1, 128)

        profiles = Profiles(axis=axis)
        profiles["a"] = 1
        profiles["b"][_next_]["c"] = 2.34
        logger.debug(profiles)
        logger.debug(type(profiles["a"]))
        logger.debug((profiles["a"]))
        logger.debug((profiles["b"][0]["c"]))


if __name__ == '__main__':
    unittest.main()
