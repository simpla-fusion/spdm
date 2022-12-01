import unittest
from typing import Any, Iterator, Mapping

from spdm.util.logger import logger
from spdm.data.AoS import AoS, SoA
from spdm.data  import Node


class TestAttributeTree(unittest.TestCase):
    def test_attribute_initialize(self):
        d = AttributeTree({
            "c": "I'm {age}!",
            "d": {
                "e": "{name} is {age}",
                "f": "{address}"
            }
        })
