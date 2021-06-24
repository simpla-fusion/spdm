import unittest

from spdm.data.Entry import Entry, EntryCombiner, EntryWrapper,  _next_, _not_found_, _undefined_
from spdm.util.logger import logger


class TestEntry(unittest.TestCase):

    def test_put(self):
        cache = {}
        d = Entry(cache)

        d.child(["a"]).push("hello world {name}!")
        d.child(["c", _next_]).push(1.23455)
        d.child(["c", _next_]).push({"a": "hello world", "b": 3.141567})

        self.assertEqual(cache["a"],                        "hello world {name}!")
        self.assertEqual(cache["c"][0],                                   1.23455)
        self.assertEqual(cache["c"][1]["a"],                        "hello world")
        self.assertEqual(cache["c"][1]["a"],                        "hello world")

        d.child(["e", "f"]).push(5)
        d.child(["e", "g"]).push(6)

        self.assertEqual(cache["e"]["f"],                                       5)
        self.assertEqual(cache["e"]["g"],                                       6)

    def test_property(self):
        cache = {
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
        d = Entry(cache)

        self.assertTrue(d.exists)
        self.assertTrue(d.child("a").exists)
        self.assertTrue(d.child("d.e").exists)
        self.assertFalse(d.child("b.f").exists)

        self.assertEqual(d.count,              3)
        self.assertEqual(d.child("a").count,   6)
        self.assertEqual(d.child("d").count,   2)

    def test_get(self):
        cache = {
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
        d = Entry(cache)

        self.assertEqual(d.get("c"),                         cache["c"])
        self.assertEqual(d.get(["d", "e"]),             cache["d"]["e"])
        self.assertEqual(d.get(["d", "f"]),             cache["d"]["f"])
        self.assertEqual(d.get(["a", 0]),                 cache["a"][0])
        self.assertEqual(d.get(["a", 1]),                 cache["a"][1])

        self.assertTrue(d.get(["a", slice(2, 6)]).equal([1, 2, 3, 4]))
        self.assertFalse(d.get("f.g").exists)

    def test_update(self):
        cache = {
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

     # def test_put(self):


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
        self.assertEqual(d.get("value"), sum([d["value"] for d in self.data]))
        self.assertEqual(d.get("d.g"), self.data[0]["d"]["g"]+self.data[2]["d"]["g"])

    def test_cache(self):
        cache = {}
        d = EntryCombiner(self.data, cache=cache)
        expected = sum([d["value"] for d in self.data])

        c = d.child("value")
        self.assertEqual(c.pull(), expected)
        self.assertEqual(cache["value"], expected)
        c.push(5)
        self.assertEqual(cache["value"], 5)

        d.child("test_cache").push("just test cache")
        self.assertEqual(d.get("test_cache", cache="off", default_value=_undefined_), _undefined_)
        self.assertEqual(d.get("test_cache", cache="on"), cache["test_cache"])


class TestEntryWrapper(unittest.TestCase):
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
        entry = Entry(self.data)
        d = EntryWrapper(entry)

    def test_property(self):
        entry = Entry(self.data)
        d = EntryWrapper(entry)

    def test_put(self):
        entry = Entry(self.data)
        d = EntryWrapper(entry)

    def test_erase(self):
        entry = Entry(self.data)
        d = EntryWrapper(entry)

    def test_cache(self):
        entry = Entry(self.data)
        d = EntryWrapper(entry)
    #     d = Entry(cache)

    #     d.child("a").put("hello world {name}!")
    #     self.assertEqual(cache["a"], "hello world {name}!")

    #     d["c"][_next_] = 1.23455
    #     d["c"][_next_] = {"a": "hello world", "b": 3.141567}

    #     self.assertEqual(cache["c"][0],  1.23455)
    # def test_append(self):
    #     d = Entry()
    #     d.child(_next_).put({"a": 1, "b": 2})

    #     self.assertEqual(d.count, 1)
    #     # self.assertTrue(d.__category__ | Entry.Category.LIST)
    #     self.assertEqual(d.child([0, "a"]).get(), 1)
    #     self.assertEqual(d.child([0, "b"]).get(), 2)


if __name__ == '__main__':
    unittest.main()
