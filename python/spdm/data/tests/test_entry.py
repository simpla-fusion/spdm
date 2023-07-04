import unittest
from copy import deepcopy

from spdm.utils.logger import logger
from spdm.utils.tags import _not_found_
from spdm.data.Entry import Entry, as_value


class TestEntry(unittest.TestCase):
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

    def test_query(self):

        d = Entry(self.data)

        self.assertEqual(d.child("c")  .__value__,           self.data["c"])
        self.assertEqual(d.child("d/e").__value__,      self.data["d"]["e"])
        self.assertEqual(d.child("d/f").__value__,      self.data["d"]["f"])
        self.assertEqual(d.child("a/0").__value__,        self.data["a"][0])
        self.assertEqual(d.child("a/1").__value__,        self.data["a"][1])

    def test_put(self):
        cache = {}

        d = Entry(cache)

        d.update({"a": "hello world {name}!"})

        self.assertEqual(cache["a"], "hello world {name}!")

        d.child("e/f").insert(5)

        d.child("e/g").insert(6)

        self.assertEqual(cache["e"]["f"],   5)

        self.assertEqual(cache["e"]["g"],   6)

    def test_exists(self):
        d = Entry(self.data)

        self.assertTrue(d.exists)
        self.assertTrue(d.child("a").exists)
        self.assertTrue(d.child("d/e").exists)
        self.assertFalse(d.child("b/h").exists)
        self.assertFalse(d.child("f/g").exists)

    def test_count(self):
        d = Entry(self.data)

        self.assertEqual(d.count,          3)
        self.assertEqual(d.child("a").count,   6)
        self.assertEqual(d.child("d").count,   2)

        # self.assertTrue(d.child(["a", slice(2, 6)]).equal([1, 2, 3, 4]))

    def test_next(self):
        data = [1, 2, 3, 4, 5]

        d = Entry(data)
        d0 = d[0]

        self.assertEqual(next(d0), data[0])
        self.assertEqual(next(d0), data[1])
        self.assertEqual(next(d0), data[2])
        self.assertEqual(next(d0), data[3])
        self.assertEqual(next(d0), data[4])
        self.assertRaises(StopIteration, next, d0)

    def test_iter(self):
        data = [1, 2, 3, 4, 5]

        d0 = Entry(data)

        self.assertListEqual([as_value(v) for v in d0], data)

    def test_insert(self):
        cache = deepcopy(self.data)
        d = Entry(cache)

        d.child("c").insert(1.23455)

        d.child("c").insert([{"a": "hello world", "b": 3.141567}])

        self.assertEqual(cache["c"][1],                      1.23455)
        self.assertEqual(cache["c"][2]["a"],           "hello world")
        self.assertEqual(cache["c"][2]["b"],                3.141567)

    def test_update(self):
        cache = deepcopy(self.data)

        d = Entry(cache)

        d.update({"d": {"g": 5}})

        self.assertEqual(cache["d"]["e"], "{name} is {age}")
        self.assertEqual(cache["d"]["f"], "{address}")
        self.assertEqual(cache["d"]["g"], 5)

    def test_erase(self):
        cache = {
            "a": [
                "hello world {name}!",
                "hello world2 {name}!",
                1, 2, 3, 4
            ],
            "b": "hello world!"
        }

        d = Entry(cache)
        del d["b"]
        self.assertTrue("b" not in cache)

    def test_get_many(self):
        cache = deepcopy(self.data)

        d = Entry(cache)

        self.assertEqual(d.child('a/2').__value__, self.data['a'][2])

        res = d.child({"a/2", "c",  "d/e", "e"}).__value__

        self.assertDictEqual(res, {"a/2": cache['a'][2],
                                   "c": cache['c'],
                                   "d/e":   cache['d']['e'],
                                   "e": _not_found_})


if __name__ == '__main__':
    unittest.main()
