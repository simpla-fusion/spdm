import unittest
from copy import deepcopy
import pprint
from spdm.data.Path import Path
from spdm.utils.logger import logger
from spdm.utils.tags import _not_found_


class TestPath(unittest.TestCase):

    def test_init(self):
        p = ['a', 'b', {'c':  slice(4, 10, 2)}]
        self.assertEqual(Path(p)[:], p)

    def test_parser(self):
        self.assertEqual(Path._parser("a/b/c"), ["a", "b", "c"])
        self.assertEqual(Path._parser("a/b/c/0"), ["a", "b", "c", 0])
        self.assertEqual(Path._parser("a/b/c/0/d"), ["a", "b", "c", 0, "d"])

        self.assertEqual(Path._parser("a/(1,2,3,'a')/h"), ["a", (1, 2, 3, 'a'), "h"])
        self.assertEqual(Path._parser("a/{1,2,3,'a'}/h"), ["a", {1, 2, 3, 'a'}, "h"])
        self.assertEqual(Path._parser("a/{'$le':[1,2]}/h"),  ["a", {Path.tags.le: [1, 2]}, "h"])
        self.assertEqual(Path._parser("a/1:10:-3/h"), ["a", slice(1, 10, -3), "h"])
        self.assertEqual(Path._parser("a/1:10:-3/$next"), ["a", slice(1, 10, -3), Path.tags.next])

    def test_append(self):
        p = Path()
        p.append('a/b/c')

        self.assertEqual(p[:], ['a', 'b', 'c'])

    data = {
        "a": [
            "hello world {name}!",
            "hello world2 {name}!",
            1, 2, 3, 4
        ],
        "c": "I'm {age}!",
        "d": {
            "e": "{name} is {age}",
            "f": "address"
        }
    }

    def test_query(self):

        self.assertEqual(Path("c").query(self.data),           self.data["c"])
        self.assertEqual(Path("d/e").query(self.data),      self.data["d"]["e"])
        self.assertEqual(Path("d/f").query(self.data),      self.data["d"]["f"])
        self.assertEqual(Path("a/0").query(self.data),        self.data["a"][0])
        self.assertEqual(Path(["a", 1]).query(self.data),        self.data["a"][1])
        # self.assertEqual(Path("a/2:4:1").query(self.data),        self.data["a"][2:4])
        self.assertEqual(Path("d/k").query(self.data, default_value=None),      None)

    def test_query_op(self):

        self.assertEqual(Path(["a"]).query(self.data, Path.tags.count), 6)
        self.assertEqual(Path(["d/e"]).query(self.data, Path.tags.count), 1)
        self.assertEqual(Path(["b/h"]).query(self.data, Path.tags.count), 0)
        self.assertEqual(Path(["d/f"]).query(self.data, Path.tags.count), 1)

        self.assertEqual(Path().query(self.data, Path.tags.count),          3)
        self.assertEqual(Path(["a"]).query(self.data, Path.tags.count),   6)
        self.assertEqual(Path(["d"]).query(self.data, Path.tags.count),   2)

        # self.assertTrue(Path(["a", slice(2, 7), {Path.tags.equal: [1, 2, 3, 4]}]).query(self.data))

    def test_insert(self):
        cache = {}

        Path("a").insert(cache,  "hello world {name}!")
        self.assertEqual(cache["a"], "hello world {name}!")

        Path("e/f").insert(cache, 5)
        Path("e/g").insert(cache, 6)

        self.assertEqual(cache["e"]["f"],   5)
        self.assertEqual(cache["e"]["g"],   6)

        cache = deepcopy(self.data)

        Path("c").insert(cache, [{"a": "hello world", "b": 3.141567}])
        Path("c").insert(cache, 1.23455)

        self.assertEqual(cache["c"][0],                 "I'm {age}!")
        self.assertEqual(cache["c"][1]["a"],           "hello world")
        self.assertEqual(cache["c"][1]["b"],                3.141567)
        self.assertEqual(cache["c"][2],                    1.23455)

    def test_update(self):
        cache = deepcopy(self.data)

        Path().update(cache, {"d": {"g": 5, "f": 6}})     

        self.assertEqual(cache["d"]["e"], "{name} is {age}")
        self.assertEqual(cache["d"]["f"], 6)
        self.assertEqual(cache["d"]["g"], 5)

    def test_delete(self):
        cache = {
            "a": [
                "hello world {name}!",
                "hello world2 {name}!",
                1, 2, 3, 4
            ],
            "b": "hello world!"
        }

        Path("b").remove(cache)

        self.assertTrue("b" not in cache)

    # def test_find_many(self):

    #     res = Path(("a/2", "c",  "d/e", "e")).query(self.data, default_value=_not_found_)

    #     self.assertListEqual(res, [self.data['a'][2],
    #                                self.data['c'],
    #                                self.data['d']['e'],
    #                                _not_found_])


if __name__ == '__main__':
    unittest.main()
