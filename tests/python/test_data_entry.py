from logging import log
import unittest
from copy import deepcopy
from spdm.data.Entry import Entry, EntryCombiner, EntryWrapper,  _next_, _not_found_, _undefined_
from spdm.util.logger import logger


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

        self.assertEqual(d.pull("c"),                self.data["c"])
        self.assertEqual(d.pull(["d", "e"]),    self.data["d"]["e"])
        self.assertEqual(d.pull(["d", "f"]),    self.data["d"]["f"])
        self.assertEqual(d.pull(["a", 0]),        self.data["a"][0])
        self.assertEqual(d.pull(["a", 1]),        self.data["a"][1])

    def test_find_by_cond(self):
        cache = [
            {"name": "wang wu", "age": 21},
            {"name": "wang liu", "age": 22},
            {"name": "li si",    "age": 22},
            {"name": "zhang san", "age": 24},
        ]

        d0 = Entry(cache)

        self.assertEqual(d0.pull(predication={"name": "li si"}, only_first=True)["age"], 22)

        self.assertEqual(d0.pull([{"name": "li si"}, "age"]), 22)

        d1 = Entry({"person": cache})

        young = d1.pull(["person", {"age": 22}])

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

        d0.push({Entry.op_tag.update: {"address": "hefei"}}, predication={"name": "wang wu"})

        self.assertEqual(cache[0]["address"],  "hefei")
        self.assertEqual(cache[0]["age"],  21)

    def test_put(self):
        cache = {}

        d = Entry(cache)

        d.push({"a": "hello world {name}!"})

        self.assertEqual(cache["a"], "hello world {name}!")

        d.push(["e", "f"], 5)

        d.push(["e", "g"], 6)

        self.assertEqual(cache["e"]["f"],   5)

        self.assertEqual(cache["e"]["g"],   6)

    def test_operator(self):
        d = Entry(self.data)

        self.assertTrue(d.pull(Entry.op_tag.exists))
        self.assertTrue(d.pull({"a": Entry.op_tag.exists}))
        self.assertTrue(d.pull({"d.e": Entry.op_tag.exists}))
        self.assertFalse(d.pull({"b.h": Entry.op_tag.exists}))
        self.assertFalse(d.pull({"f.g": Entry.op_tag.exists}))

        self.assertEqual(d.pull(Entry.op_tag.count),          3)
        self.assertEqual(d.pull({"a": Entry.op_tag.count}),   6)
        self.assertEqual(d.pull({"d": Entry.op_tag.count}),   2)

        self.assertTrue(d.pull(["a", slice(2, 6)], {Entry.op_tag.equal: [1, 2, 3, 4]}))

    def test_append(self):
        cache = {}
        d = Entry(cache)
        d.push(["c", _next_],  1.23455)
        d.push(["c", _next_], {"a": "hello world", "b": 3.141567})

        self.assertEqual(cache["c"][0],                      1.23455)
        self.assertEqual(cache["c"][1]["a"],           "hello world")
        self.assertEqual(cache["c"][1]["b"],                3.141567)

    def test_update(self):
        cache = deepcopy(self.data)

        d = Entry(cache)

        d.push({"d": {"g": 5}})

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
        d.remove("b")
        self.assertTrue("b" not in cache)


class TestEntryCombiner(unittest.TestCase):
    data = [
        {"id": 0,
            "value": 1.23,
            "c": "I'm {age}!",
            "d": {"e": "{name} is {age}", "f": "{address}", "g": [1, 2, 3]}},
        {"id": 1,
            "value": 2.23,
            "c": "I'm {age}!",
            "d": {"e": "{name} is {age}", "f": "{address}"}},
        {"id": 2,
            "value": 4.23,
            "c": "I'm {age}!",
            "d": {"e": "{name} is {age}", "f": "{address}", "g": [4, 5, 7]}}
    ]

    def test_get(self):
        d = EntryCombiner(self.data)
        self.assertEqual(d.pull("value"), sum([d["value"] for d in self.data]))
        self.assertEqual(d.pull("d.g"), self.data[0]["d"]["g"]+self.data[2]["d"]["g"])

    # def test_cache(self):
    #     cache = {}
    #     d = EntryCombiner(self.data )
    #     expected = sum([d["value"] for d in self.data])

    #     c = d.extend("value")
    #     self.assertEqual(c.pull(), expected)
    #     self.assertEqual(cache["value"], expected)
    #     c.push(5)
    #     self.assertEqual(cache["value"], 5)

    #     d.extend("test_cache").push("just test cache")
    #     self.assertEqual(d.get("test_cache",default_value=_undefined_), _undefined_)
    #     self.assertEqual(d.get("test_cache"), cache["test_cache"])

# class TestEntryWrapper(unittest.TestCase):
#     data = [
#         {"id": 0,
#          "value": 1.23,
#          "c": "I'm {age}!",
#          "d": {"e": "{name} is {age}", "f": "{address}", "g": [1, 2, 3]}},
#         {"id": 1,
#          "value": 2.23,
#          "c": "I'm {age}!",
#          "d": {"e": "{name} is {age}", "f": "{address}"}},
#         {"id": 2,
#          "value": 4.23,
#          "c": "I'm {age}!",
#          "d": {"e": "{name} is {age}", "f": "{address}", "g": [4, 5, 7]}}
#     ]

#     def test_get(self):
#         entry = Entry(self.data)
#         d = EntryWrapper(entry)

#     def test_property(self):
#         entry = Entry(self.data)
#         d = EntryWrapper(entry)

#     def test_put(self):
#         entry = Entry(self.data)
#         d = EntryWrapper(entry)

#     def test_erase(self):
#         entry = Entry(self.data)
#         d = EntryWrapper(entry)

#     def test_cache(self):
#         entry = Entry(self.data)
#         d = EntryWrapper(entry)
#     #     d = Entry(cache)

#     #     d.extend("a").put("hello world {name}!")
#     #     self.assertEqual(cache["a"], "hello world {name}!")

#     #     d["c"][_next_] = 1.23455
#     #     d["c"][_next_] = {"a": "hello world", "b": 3.141567}

#     #     self.assertEqual(cache["c"][0],  1.23455)
#     # def test_append(self):
#     #     d = Entry()
#     #     d.extend(_next_).put({"a": 1, "b": 2})

#     #     self.assertEqual(d.count, 1)
#     #     # self.assertTrue(d.__category__ | Entry.Category.LIST)
#     #     self.assertEqual(d.extend([0, "a"]).get(), 1)
#     #     self.assertEqual(d.extend([0, "b"]).get(), 2)


if __name__ == '__main__':
    unittest.main()
