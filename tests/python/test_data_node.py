import importlib
import pprint
import sys
import unittest
from spdm.data.Node import Node


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


if __name__ == '__main__':

    unittest.main()
