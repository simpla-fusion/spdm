import unittest
from copy import deepcopy
from logging import log

from spdm.common.logger import logger
from spdm.common.tags import _not_found_
from spdm.data.Entry import Entry, EntryCombine
from spdm.data.Path import Path
from spdm.data.Query import Query


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

        self.assertEqual(d.child("c").pull(),           self.data["c"])
        self.assertEqual(d.child("d", "e").pull(),      self.data["d"]["e"])
        self.assertEqual(d.child("d", "f").pull(),      self.data["d"]["f"])
        self.assertEqual(d.child("a", 0).pull(),        self.data["a"][0])
        self.assertEqual(d.child("a", 1).pull(),        self.data["a"][1])

    def test_find_by_query(self):
        cache = [
            {"name": "wang wu", "age": 21},
            {"name": "wang liu", "age": 22},
            {"name": "li si",    "age": 22},
            {"name": "Zhang san", "age": 24},
        ]

        d0 = Entry(cache)

        # self.assertEqual(d0.pull(predication={"name": "li si"}, only_first=True)["age"], 22)

        self.assertEqual(d0.child(Query({"name": "li si"}), "age").pull(), 22)

        d1 = Entry({"person": cache})

        young = d1.child("person", Query({"age": 22}, only_first=False)).pull()

        self.assertEqual(len(young), 2)
        self.assertEqual(young[0]["name"],  "wang liu")
        self.assertEqual(young[1]["name"],  "li si")

    def test_update_by_cond(self):
        cache = [
            {"name": "wang wu",   "age": 21},
            {"name": "wang liu",  "age": 22},
            {"name": "li si",     "age": 22},
            {"name": "zhang san", "age": 24},
        ]

        d0 = Entry(cache)

        d0.child(Query({"name": "wang wu"}), "address").push("hefei")

        self.assertEqual(cache[0]["address"],  "hefei")

        self.assertEqual(cache[0]["age"],  21)

    def test_put(self):
        cache = {}

        d = Entry(cache)

        d.update({"a": "hello world {name}!"})

        self.assertEqual(cache["a"], "hello world {name}!")

        d.child("e.f").push(5)

        d.child("e.g").push(6)

        self.assertEqual(cache["e"]["f"],   5)

        self.assertEqual(cache["e"]["g"],   6)

    def test_operator(self):
        d = Entry(self.data)

        self.assertTrue(d.exists())
        self.assertTrue(d.child("a").exists())
        self.assertTrue(d.child("d.e").exists())
        self.assertFalse(d.child("b.h").exists())
        self.assertFalse(d.child("f.g").exists())

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
        d.child("b").erase()
        self.assertTrue("b" not in cache)

    def test_get_many(self):
        d = Entry(self.data)
        res = d.child([Path("a", 2), "c", Path("d.e"), "e"]).pull()
        self.assertListEqual(res, [1, "I'm {age}!", "{name} is {age}",  _not_found_])


if __name__ == '__main__':
    unittest.main()
