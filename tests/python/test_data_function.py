import logging
import pprint
import sys
import unittest

import numpy as np
from spdm.data.Function2 import Function
from spdm.util.logger import logger


class TestFunction(unittest.TestCase):
    def test_init(self):
        x = np.linspace(0, 1, 128)
        
        fun = Function(x, 1.0)

        logger.debug(fun)

    def test_operator(self):
        x = np.linspace(0, 1, 128)

        fun = Function(x, 1.0)

        logger.debug(dir(fun))

        logger.debug(type(np.sin(fun)))
        logger.debug(-fun)
        logger.debug(fun + 2)
        logger.debug(fun - 2)
        logger.debug(fun * 2)
        logger.debug(fun / 2)
        logger.debug(fun ** 2)
        # logger.debug(fun @ fun)


if __name__ == '__main__':
    unittest.main()
