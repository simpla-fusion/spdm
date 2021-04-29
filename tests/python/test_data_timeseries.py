import importlib
import pprint
import sys
import unittest

import numpy as np
from spdm.data.Node import Node, _next_
from spdm.data.Profiles import Profiles
from spdm.data.TimeSeries import TimeSeries, TimeSlice
from spdm.util.logger import logger


class TestTimeSeries(unittest.TestCase):
    def test_timeseries_initialize(self):
        cache = [{"a": 1},
                 {"a": 2},
                 {"a": 3}, ]

        class Foo(Node):
            __slot__ = ("_time")

            def __init__(self,  *args, time=None, **kwargs):
                super().__init__(*args, **kwargs)
                self._time = time or 0.0

            def __repr__(self) -> str:
                return f"<{self.__class__.__name__} time='{self._time}' />"

        time_series = TimeSeries[Foo](cache, time=None, dt=0.1)

        time_series[_next_] = {"a": 4}

        logger.debug(time_series)

        logger.debug(time_series[-1])


if __name__ == '__main__':
    unittest.main()
