import importlib
import logging
import pprint
import sys
import unittest
from spdm.data.Node import Node
from spdm.util.logger import logger


class TestNode(unittest.TestCase):

    def test_attribute_travel(self):
        cache = {
            "a": [
                "hello world {name}!",
                "hello world2 {name}!",
                1, 2, 3, 4
            ],
            "b": {
                "c": "I'm {age}!",
                "d": {
                    "e": "{name} is {age}",
                    "f": "{address}"
                }
            }
        }
        d = Node(cache)
        d["d"] = 5

        pprint.pprint(cache)

    def test_node_iter(self):
        d = Node([1, 2, 3, 4, 5, 6])
        self.assertEqual([v for v in d],  [1, 2, 3, 4, 5, 6])

    def test_node_expend_dict(self):
        d = Node({"a": {"b": 1, "c": 2}})
        logger.debug({**d["a"]})


if __name__ == '__main__':

    unittest.main()
