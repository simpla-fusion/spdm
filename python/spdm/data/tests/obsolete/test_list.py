import unittest
from copy import copy, deepcopy

import numpy as np
from spdm.data.Dict import Dict
from spdm.data.List import List
from spdm.data.HTree import HTree
from spdm.data.Path import Path
from spdm.utils.logger import logger


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
        d["c"].append(1.23455)
        d["c"].append({"a": "hello world", "b": 3.141567})
        self.assertEqual(cache["a"], "hello world {name}!")
        self.assertEqual(cache["c"][0],  1.23455)
        self.assertEqual(cache["c"][1]["a"], "hello world")
        self.assertEqual(d["c"][1]["a"].value, cache["c"][1]["a"])

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
        self.assertEqual(d0[({"name": "li si"}), "age"].value, 22)

        d1 = Dict({"person": cache})

        young = d1["person",  {"age": 22}]

        self.assertEqual(young[0, "name"].value,  "wang liu")
        self.assertEqual(young[1, "name"].value,  "li si")

        # res = d1["person", {"age": 22}]
        # names = [d["name"] for d in res]
        # self.assertEqual(len(names), 2)

    def test_insert_by_cond(self):
        cache = [
            {"name": "wang wu",   "age": 21},
            {"name": "wang liu",  "age": 22},
            {"name": "li si",     "age": 22},
            {"name": "zhang san", "age": 24},
        ]

        d0 = List(cache)

        d0[{"name": "wang wu"}, "address"] = "hefei"

        self.assertEqual(cache[0]["address"],  "hefei")

    def test_iter(self):
        d = List(self.data["a"])

        expected = [v for v in d]
        self.assertEqual(self.data["a"], expected)


if __name__ == '__main__':
    unittest.main()
