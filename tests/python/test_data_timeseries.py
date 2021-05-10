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
                 {"a": 2, "time": 5},
                 {"a": 3}, ]

        class Foo(TimeSlice):
            def __init__(self,  *args,   **kwargs):
                super().__init__(*args, **kwargs)

            def __repr__(self) -> str:
                return f"<{self.__class__.__name__} time='{self.time}' />"

        time_series = TimeSeries[Foo](cache, time_start=0.2, time_step=0.1)

        time_series[_next_] = {"a": 4}

        logger.debug(time_series)

        time_series.sort()

        logger.debug(time_series)

        time_series.insert({"a": 10, "time": 0.45})

        logger.debug(time_series)

        time_series.insert({"a": 12})
        
        logger.debug(time_series)


if __name__ == '__main__':
    unittest.main()
