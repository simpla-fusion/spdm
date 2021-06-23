import unittest

from spdm.data.Node import Dict, List, Node, _next_, _not_found_
from spdm.util.logger import logger


class TestNodeList(unittest.TestCase):
    def test_insert(self):
        cache = {}

        d = Dict(cache)

        d["a"] = "hello world {name}!"
        d["c"][_next_] = 1.23455
        d["c"][_next_] = {"a": "hello world", "b": 3.141567}
        
        self.assertEqual(cache["a"], "hello world {name}!")
        self.assertEqual(cache["c"][0],  1.23455)
        self.assertEqual(cache["c"][1]["a"], "hello world")
        self.assertEqual(d["c"][1]["a"], "hello world")

        d["e"]["f"] = 5
        d["e"]["g"] = 6
        self.assertEqual(cache["e"]["f"], 5)
        self.assertEqual(cache["e"]["g"], 6)

    def test_list_find_by_cond(self):
        cache = [
            {"name": "wang wu", "age": 21},
            {"name": "wang liu", "age": 22},
            {"name": "li si",    "age": 22},
            {"name": "zhang san", "age": 24},
        ]

        d0 = List(cache)
        self.assertEqual(d0[{"name": "li si"}]["age"], 22)

        d1 = Dict({"person": cache})

        young = d1.find(["person", {"age": 22}])

        self.assertEqual(len(young), 2)
        self.assertEqual(young[0]["name"],  "wang liu")
        self.assertEqual(young[1]["name"],  "li si")

        res = d1["person", {"age": 22}]

        logger.debug(type(res))

        names = [d["name"] for d in res]

        self.assertEqual(len(names), 2)

    def test_list_insert_by_cond(self):
        cache = [
            {"name": "wang wu",   "age": 21},
            {"name": "wang liu",  "age": 22},
            {"name": "li si",     "age": 22},
            {"name": "zhang san", "age": 24},
        ]

        d0 = List(cache)

        d0[{"name": "wang wu"}]["address"] = "hefei"

        self.assertEqual(cache[0]["address"],  "hefei")


if __name__ == '__main__':
    unittest.main()
