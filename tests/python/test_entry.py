import unittest
from copy import deepcopy

from spdm.common.tags import _not_found_
from spdm.data.Entry import Entry


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

    def test_get(self):

        d = Entry(self.data)

        self.assertEqual(d.get("c"),           self.data["c"])
        self.assertEqual(d.get("d/e"),      self.data["d"]["e"])
        self.assertEqual(d.get("d/f"),      self.data["d"]["f"])
        self.assertEqual(d.get("a/0"),        self.data["a"][0])
        self.assertEqual(d.get("a/1"),        self.data["a"][1])

    def test_put(self):
        cache = {}

        d = Entry(cache)

        d.update({"a": "hello world {name}!"})

        self.assertEqual(cache["a"], "hello world {name}!")

        d.put("e/f", 5)

        d.put("e/g", 6)

        self.assertEqual(cache["e"]["f"],   5)

        self.assertEqual(cache["e"]["g"],   6)

    def test_operator(self):
        d = Entry(self.data)

        self.assertTrue(d.exists())
        self.assertTrue(d.child("a").exists())
        self.assertTrue(d.child("d/e").exists())
        self.assertFalse(d.child("b/h").exists())
        self.assertFalse(d.child("f/g").exists())

        self.assertEqual(d.count(),          3)
        self.assertEqual(d.child("a").count(),   6)
        self.assertEqual(d.child("d").count(),   2)

        # self.assertTrue(d.child(["a", slice(2, 6)]).equal([1, 2, 3, 4]))

    def test_append(self):
        cache = {}
        d = Entry(cache)

        d.child("c").append(1.23455)

        d.child("c").extend([{"a": "hello world", "b": 3.141567}])

        self.assertEqual(cache["c"][0],                      1.23455)
        self.assertEqual(cache["c"][1]["a"],           "hello world")
        self.assertEqual(cache["c"][1]["b"],                3.141567)

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
        d = Entry(self.data)

        self.assertEqual(d.get('a/2'), self.data['a'][2])

        res = d.get(["a/2", "c",  "d/e", "e"])
        self.assertListEqual(res, [self.data['a'][2],
                                   self.data['c'],
                                   self.data['d']['e'],
                                   _not_found_])


if __name__ == '__main__':
    unittest.main()
