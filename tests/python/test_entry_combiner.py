
class TestEntryCombiner(unittest.TestCase):
    data = [
        {"id": 0,
            "value": 1.23,
            "c": "I'm {age}!",
            "d": {"e": "{name} is {age}", "f": "{address}", "g": [1, 2, 3]}},
        {"id": 1,
            "c": "I'm {age}!",
            "d": {"e": "{name} is {age}", "f": "{address}"}},
        {"id": 2,
            "value": 4.23,
            "c": "I'm {age}!",
            "d": {"e": "{name} is {age}", "f": "{address}", "g": [4, 5, 7]}}
    ]

    def test_get(self):
        d = EntryCombiner(self.data)
        self.assertEqual(d.pull(), sum([v.get("value", 0.0) for v in self.data]))
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
