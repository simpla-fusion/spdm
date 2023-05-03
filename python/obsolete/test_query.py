import unittest
from copy import deepcopy
from logging import log

from spdm.utils.logger import logger
from spdm.utils.tags import _not_found_
from spdm.data.Entry import Entry, EntryCombine
from spdm.data.Path import Path


class TestQuery(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
