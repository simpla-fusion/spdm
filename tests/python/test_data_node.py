import unittest

from spdm.data.Node import Node, Dict, List, _next_, _not_found_
from spdm.util.logger import logger


class TestNode(unittest.TestCase):
    def test_dict_initialize(self):
        d = Dict({
            "c": "I'm {age}!",
            "d": {
                "e": "{name} is {age}",
                "f": "{address}"
            }
        })

    def test_dict_find_by_key(self):
        cache = {
            "a": [
                "hello world {name}!",
                "hello world2 {name}!",
                1, 2, 3, 4
            ],
            "c": "I'm {age}!",
            "d": {
                "e": "{name} is {age}",
                "f": "{address}"
            }
        }
        d = Dict(cache)

        self.assertEqual(len(d["a"]),                 6)
        self.assertEqual(d["c"],             cache["c"])
        self.assertEqual(d["d"]["e"],   cache["d"]["e"])
        self.assertEqual(d["d"]["f"],   cache["d"]["f"])
        self.assertEqual(d["a"][0],       cache["a"][0])
        self.assertEqual(d["a"][1],       cache["a"][1])
        self.assertEqual(d["a"][2:6],      [1, 2, 3, 4])

        # self.assertTrue(d["f"]["g"].empty)

    def test_dict_insert(self):
        cache = {}

        d = Dict(cache)

        d["a"] = "hello world {name}!"

        self.assertEqual(cache["a"], "hello world {name}!")

        d["e"]["f"] = 5
        d["e"]["g"] = 6
        self.assertEqual(cache["e"]["f"], 5)
        self.assertEqual(cache["e"]["g"], 6)

    def test_dict_update(self):
        cache = {
            "a": [
                "hello world {name}!",
                "hello world2 {name}!",
                1, 2, 3, 4
            ],
            "c": "I'm {age}!",
            "d": {
                "e": "{name} is {age}",
                "f": "{address}"
            }
        }
        d = Dict(cache)

        d.update({"d": {"g": 5}})

        self.assertEqual(cache["d"]["e"], "{name} is {age}")
        self.assertEqual(cache["d"]["f"], "{address}")
        self.assertEqual(cache["d"]["g"], 5)

    def test_node_insert(self):
        cache = {"this_is_a_cache": True}

        d = Dict(cache)

        d["a"] = "hello world {name}!"
        self.assertEqual(cache["a"], "hello world {name}!")

        d["c"][_next_] = 1.23455
        d["c"][_next_] = {"a": "hello world", "b": 3.141567}

        self.assertEqual(cache["c"][0],  1.23455)

    def test_node_append(self):
        d = List()
        d[_next_] = {"a": 1, "b": 2}

        self.assertEqual(len(d), 1)
        self.assertTrue(d.__category__ | Node.Category.LIST)
        self.assertEqual(d[0]["a"], 1)
        self.assertEqual(d[0]["b"], 2)

    def test_node_boolean(self):
        d = Dict()
        self.assertTrue(d.empty)
        self.assertTrue(d["a"] or 12.3, 12.3)

    def test_node_del(self):
        cache = {
            "a": [
                "hello world {name}!",
                "hello world2 {name}!",
                1, 2, 3, 4
            ]
        }

        d = Node(cache)
        del d["a"]
        self.assertTrue("a" not in cache)


if __name__ == '__main__':
    unittest.main()
