import unittest

from spdm.data.Node import Node, Dict, List, _next_, _not_found_
from spdm.util.logger import logger
from copy import copy, deepcopy


class TestNode(unittest.TestCase):
    data = {
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

    def test_find_by_key(self):

        d = Dict(self.data)

        self.assertEqual(len(d["a"]),                     6)
        self.assertEqual(d["c"],             self.data["c"])
        self.assertEqual(d["d"]["e"],   self.data["d"]["e"])
        self.assertEqual(d["d"]["f"],   self.data["d"]["f"])
        self.assertEqual(d["a"][0],       self.data["a"][0])
        self.assertEqual(d["a"][1],       self.data["a"][1])
        self.assertEqual(d["a"][2:6],          [1, 2, 3, 4])

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
        cache = deepcopy(self.data)
        d = Dict(cache)

        d.update({"d": {"g": 5}})

        self.assertEqual(cache["d"]["e"], "{name} is {age}")
        self.assertEqual(cache["d"]["f"], "{address}")
        self.assertEqual(cache["d"]["g"], 5)

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

    def test_node_append(self):
        d = List()
        d[_next_] = {"a": 1, "b": 2}

        self.assertEqual(len(d), 1)
        self.assertEqual(d[0]["a"], 1)
        self.assertEqual(d[0]["b"], 2)

    def test_node_insert(self):
        cache = {"this_is_a_cache": True}

        d = Dict(cache)

        d["a"] = "hello world {name}!"
        self.assertEqual(cache["a"], "hello world {name}!")

        d["c"][_next_] = 1.23455
        d["c"][_next_] = {"a": "hello world", "b": 3.141567}

        self.assertEqual(cache["c"][0],  1.23455)

    # def test_node_boolean(self):
    #     d = Dict()
    #     self.assertTrue(d.empty)
    #     self.assertTrue(d["a"] or 12.3, 12.3)


class TestNodeList(unittest.TestCase):
    data = {
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

    def test_insert(self):
        cache = {}

        d = Dict(cache)

        d["a"] = "hello world {name}!"
        d["c", _next_] = 1.23455
        d["c", _next_] = {"a": "hello world", "b": 3.141567}

        self.assertEqual(cache["a"], "hello world {name}!")
        self.assertEqual(cache["c"][0],  1.23455)
        self.assertEqual(cache["c"][1]["a"], "hello world")
        self.assertEqual(d["c"][1]["a"], "hello world")

        d["e"]["f"] = 5
        d["e"]["g"] = 6
        self.assertEqual(cache["e"]["f"], 5)
        self.assertEqual(cache["e"]["g"], 6)

    def test_find_by_cond(self):
        cache = [
            {"name": "wang wu", "age": 21},
            {"name": "wang liu", "age": 22},
            {"name": "li si",    "age": 22},
            {"name": "zhang san", "age": 24},
        ]

        d0 = List(cache)
        self.assertEqual(d0[{"name": "li si"}, "age"], 22)

        d1 = Dict({"person": cache})

        young = d1["person", {"age": 22}]
        # logger.debug(young)
        # self.assertEqual(len(young), 2)
        self.assertEqual(young[0, "name"],  "wang liu")
        self.assertEqual(young[1, "name"],  "li si")

        res = d1["person", {"age": 22}]

        names = [d["name"] for d in res]

        self.assertEqual(len(names), 2)

    def test_insert_by_cond(self):
        cache = [
            {"name": "wang wu",   "age": 21},
            {"name": "wang liu",  "age": 22},
            {"name": "li si",     "age": 22},
            {"name": "zhang san", "age": 24},
        ]

        d0 = List(cache)

        d0[{"name": "wang wu"}] = {"address": "hefei"}

        self.assertEqual(cache[0]["address"],  "hefei")

    def test_iter(self):
        d = List(self.data["a"])

        expected = [v for v in d]
        self.assertEqual(self.data["a"], expected)


if __name__ == '__main__':
    unittest.main()
