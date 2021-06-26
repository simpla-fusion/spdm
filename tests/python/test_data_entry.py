import unittest
from copy import copy
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

    def test_put(self):
        cache = {}

        d = Entry(cache)

        d.extend(["a"]).push("hello world {name}!")

        self.assertEqual(cache["a"],           "hello world {name}!")

        d.extend(["e", "f"]).push(5)

        d.extend(["e", "g"]).push(6)

        self.assertEqual(cache["e"]["f"],   5)

        self.assertEqual(cache["e"]["g"],   6)

    def test_get(self):

        d = Entry(self.data)

        self.assertEqual(d.get("c"),                self.data["c"])
        self.assertEqual(d.get(["d", "e"]),    self.data["d"]["e"])
        self.assertEqual(d.get(["d", "f"]),    self.data["d"]["f"])
        self.assertEqual(d.get(["a", 0]),        self.data["a"][0])
        self.assertEqual(d.get(["a", 1]),        self.data["a"][1])

    def test_operator(self):
        d = Entry(self.data)

        self.assertTrue(d.exists)
        self.assertTrue(d.extend("a").exists)
        self.assertTrue(d.extend("d.e").exists)
        self.assertFalse(d.extend("b.h").exists)

        self.assertEqual(d.count,              3)
        self.assertEqual(d.extend("a").count,   6)
        self.assertEqual(d.extend("d").count,   2)

        self.assertTrue(d.extend(["a", slice(2, 6)]).equal([1, 2, 3, 4]))
        self.assertFalse(d.extend("f.g").exists)

    def test_append(self):
        cache = {}
        d = Entry(cache)
        d.put("c", {Entry.op_tag.append: 1.23455})
        d.put("c", {Entry.op_tag.append: {"a": "hello world", "b": 3.141567}})

        self.assertEqual(cache["c"][0],                      1.23455)
        self.assertEqual(cache["c"][1]["a"],           "hello world")
        self.assertEqual(cache["c"][1]["a"],           "hello world")

    def test_update(self):
        cache = copy(self.data)

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
        d.extend("b").erase()
        self.assertTrue("b" not in cache)

    def test_append(self):
        cache = {"this_is_a_cache": True}

        d = Entry(cache)

        d.put(["a"], "hello world {name}!")
        d.put(["c", _next_], 1.23455)
        d.put(["c", _next_],  "a")

        self.assertEqual(cache["a"], "hello world {name}!")
        self.assertEqual(cache["c"][0],  1.23455)
        self.assertEqual(cache["c"][1],  "a")

    def test_list_find_by_cond(self):
        cache = [
            {"name": "wang wu", "age": 21},
            {"name": "wang liu", "age": 22},
            {"name": "li si",    "age": 22},
            {"name": "zhang san", "age": 24},
        ]

        d0 = Entry(cache)
        self.assertEqual(d0.get([{"name": "li si"}, "age"]), 22)

        d1 = Entry({"person": cache})

        young = d1.get(["person", {"age": 22}], lazy=True)

        self.assertEqual(len(young), 2)
        self.assertEqual(young[0]["name"],  "wang liu")
        self.assertEqual(young[1]["name"],  "li si")

        res=d1.get(["person", {"age": 22}])

        names=[d["name"] for d in res]

        self.assertEqual(len(names), 2)


class TestEntryCombiner(unittest.TestCase):
    data=[
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
        self.assertEqual(d.get("value"), sum([d["value"] for d in self.data]))
        self.assertEqual(d.get("d.g"), self.data[0]["d"]["g"]+self.data[2]["d"]["g"])

    def test_cache(self):
        cache = {}
        d = EntryCombiner(self.data, cache=cache)
        expected = sum([d["value"] for d in self.data])

        c = d.extend("value")
        self.assertEqual(c.pull(), expected)
        self.assertEqual(cache["value"], expected)
        c.push(5)
        self.assertEqual(cache["value"], 5)

        d.extend("test_cache").push("just test cache")
        self.assertEqual(d.get("test_cache", cache="off", default_value=_undefined_), _undefined_)
        self.assertEqual(d.get("test_cache", cache="on"), cache["test_cache"])

    #     # class TestEntryWrapper(unittest.TestCase):
    #     #     data = [
    #     #         {"id": 0,
    #     #          "value": 1.23,
    #     #          "c": "I'm {age}!",
    #     #          "d": {"e": "{name} is {age}", "f": "{address}", "g": [1, 2, 3]}},
    #     #         {"id": 1,
    #     #          "value": 2.23,
    #     #          "c": "I'm {age}!",
    #     #          "d": {"e": "{name} is {age}", "f": "{address}"}},
    #     #         {"id": 2,
    #     #          "value": 4.23,
    #     #          "c": "I'm {age}!",
    #     #          "d": {"e": "{name} is {age}", "f": "{address}", "g": [4, 5, 7]}}
    #     #     ]

    #     #     def test_get(self):
    #     #         entry = Entry(self.data)
    #     #         d = EntryWrapper(entry)

    #     #     def test_property(self):
    #     #         entry = Entry(self.data)
    #     #         d = EntryWrapper(entry)

    #     #     def test_put(self):
    #     #         entry = Entry(self.data)
    #     #         d = EntryWrapper(entry)

    #     #     def test_erase(self):
    #     #         entry = Entry(self.data)
    #     #         d = EntryWrapper(entry)

    #     #     def test_cache(self):
    #     #         entry = Entry(self.data)
    #     #         d = EntryWrapper(entry)
    #     #     #     d = Entry(cache)

    #     #     #     d.extend("a").put("hello world {name}!")
    #     #     #     self.assertEqual(cache["a"], "hello world {name}!")

    #     #     #     d["c"][_next_] = 1.23455
    #     #     #     d["c"][_next_] = {"a": "hello world", "b": 3.141567}

    #     #     #     self.assertEqual(cache["c"][0],  1.23455)
    #     #     # def test_append(self):
    #     #     #     d = Entry()
    #     #     #     d.extend(_next_).put({"a": 1, "b": 2})

    #     #     #     self.assertEqual(d.count, 1)
    #     #     #     # self.assertTrue(d.__category__ | Entry.Category.LIST)
    #     #     #     self.assertEqual(d.extend([0, "a"]).get(), 1)
    #     #     #     self.assertEqual(d.extend([0, "b"]).get(), 2)


if __name__ == '__main__':
    unittest.main()
