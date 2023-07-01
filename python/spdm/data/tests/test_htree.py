import typing
import unittest
from copy import deepcopy

import numpy as np
from spdm.data.HTree import Dict, HTree, List
from spdm.utils.logger import logger


class Foo(Dict):
    def __init__(self,   *args, **kwargs):
        super().__init__(*args, **kwargs)


test_data = {
    "a": [
        "hello world {name}!",
        "hello world2 {name}!",
        1.0, 2, 3, 4
    ],
    "c": "I'm {age}!",
    "d": {
        "e": "{name} is {age}",
        "f": "{address}"
    }
}


class NamedFoo(Dict):
    a: List[typing.Any]
    c: str
    d: Dict


class TestNode(unittest.TestCase):
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
    # def test_new(self):
    #     self.assertTrue(isinstance(HTree("hello"), HTree))
    #     self.assertTrue(isinstance(HTree(1), HTree))
    #     self.assertTrue(isinstance(HTree(np.ones([10, 20])), HTree))
    #     self.assertTrue(isinstance(HTree([1, 2, 3, 4, 5]), List))
    #     self.assertTrue(isinstance(HTree((1, 2, 3, 4, 5)), List))
    #     self.assertTrue(isinstance(HTree({"a": 1, "b": 2, "c": 3}), Dict))
    #     # self.assertFalse(isinstance(HTree({1, 2, 3, 4, 5}), List))

    # def test_create(self):
    #     cache = []
    #     d = HTree(cache)
    #     self.assertEqual(d.create_child("hello"), "hello")
    #     self.assertEqual(d.create_child(1), 1)
    #     v = np.ones([10, 20])
    #     self.assertIs(d.create_child(v), v)
    #     self.assertTrue(isinstance(d.create_child("hello", always_node=True), Node))

    #     self.assertTrue(isinstance(d.create_child([1, 2, 3, 4, 5]), List))
    #     self.assertTrue(isinstance(d.create_child((1, 2, 3, 4, 5)), List))
    #     self.assertTrue(isinstance(d.create_child({"a": 1, "b": 2, "c": 3}), Dict))

    def test_get_by_path(self):
        cache = deepcopy(self.data)

        d = Dict(cache)

        self.assertEqual(len(d["a"]),                     6)
        self.assertEqual(d["c"].__value__,                 cache["c"])
        self.assertEqual(d["d/e"].__value__,          cache["d"]["e"])
        self.assertEqual(d["d/f"].__value__,          cache["d"]["f"])
        self.assertEqual(d["a/0"].__value__,            cache["a"][0])
        self.assertEqual(d["a/1"].__value__,            cache["a"][1])

        # self.assertListEqual(list(d["a"][2:6]),       [1.0, 2, 3, 4])

    def test_assign(self):
        cache = {}

        d = Dict(cache)

        d["a"] = "hello world {name}!"

        self.assertEqual(cache["a"], "hello world {name}!")
        d["e"] = {}
        d["e"]["f"] = 5
        d["e"]["g"] = 6
        self.assertEqual(cache["e"]["f"], 5)
        self.assertEqual(cache["e"]["g"], 6)

    def test_update(self):
        cache = deepcopy(test_data)
        d = Dict(cache)

        d._update([], {"d": {"g": 5}})

        self.assertEqual(cache["d"]["e"], "{name} is {age}")
        self.assertEqual(cache["d"]["f"], "{address}")
        self.assertEqual(cache["d"]["g"], 5)

    def test_insert(self):
        cache0 = deepcopy(test_data)
        d0 = Dict(cache0)

        d0._insert("a", "hello world {name}!")
        d0.insert({"d": {"g": 5}})

        self.assertEqual(cache0["d"]["e"], "{name} is {age}")
        self.assertEqual(cache0["d"]["f"], "{address}")
        self.assertEqual(cache0["d"]["g"], 5)

        cache1 = []

        d1 = List(cache1)

        d1.insert({"a": [1], "b": 2})

        self.assertEqual(cache1[0]["a"][0], 1)
        self.assertEqual(cache1[0]["b"], 2)

        d1._insert("0/a", 2)

        self.assertEqual(cache1[0]["a"], [1, 2])

    def test_get_by_index(self):
        data = [1, 2, 3, 4, 5]

        d0 = List[int](data)
        # logger.debug(type(d0[0]))
        self.assertIsInstance(d0[0], int)
        self.assertEqual(d0[0], data[0])
        # self.assertListEqual(list(d0[:]), data)

    def test_node_del(self):
        cache = {
            "a": [
                "hello world {name}!",
                "hello world2 {name}!",
                1, 2, 3, 4
            ]
        }

        d = Dict(cache)

        del d["a"]

        self.assertTrue("a" not in cache)

    def test_node_insert(self):
        cache = {"this_is_a_cache": True}

        d = Dict[List](cache)

        d["a"] = "hello world {name}!"
        self.assertEqual(cache["a"], "hello world {name}!")

        d["c"] .insert(1.23455)
        d["c"] .insert({"a": "hello world", "b": 3.141567})

        self.assertEqual(cache["c"][0],  1.23455)
        self.assertEqual(cache["c"][1]["b"],  3.141567)

    def test_type_hint(self):
        d1 = List([])
        d1 .insert({"a": 1, "b": 2})

        self.assertIsInstance(d1[0], HTree)

        self.assertEqual(len(d1), 1)
        self.assertEqual(d1[0]["a"].__value__, 1)
        self.assertEqual(d1[0]["b"].__value__, 2)

        data = [1, 2, 3, 4, 5]

        class Foo:
            def __init__(self, v, **kwargs) -> None:
                self.v = v

        d1 = List[Foo](data)

        self.assertIsInstance(d1[2], Foo)
        self.assertEqual(d1[2].v.__value__, data[2])

    def test_get_by_slice(self):
        data = [1, 2, 3, 4, 5]

        d0 = List[int](data)

        self.assertListEqual(d0[:].__value__, data)

    # def test_child_type_convert_list(self):

    #     cache = [{"a": 1234}, {"b": 1234}, {"c": 1234}, {"d": 1234}]

    #     d = List[Foo](cache)

    #     self.assertFalse(isinstance(cache[1], Foo))
    #     self.assertTrue(isinstance(d[1], Foo))

    #     self.assertTrue(isinstance(cache[1], Foo))

    # def test_chain_mapping(self):
    #     cache = {"a": 1234, "b": 1234, "c": 12343, "d": 12345}
    #     d = Dict(cache, a=5, b=4)
    #     self.assertEqual(d["a"], 5)
    #     self.assertEqual(d["b"], 4)

    #     self.assertEqual(d["c"], cache["c"])
    #     self.assertEqual(d["d"], cache["d"])

    # def test_node_boolean(self):
    #     d = Dict()
    #     self.assertTrue(d.empty)
    #     self.assertTrue(d["a"] or 12.3, 12.3)

    # def test_dict_find_by_key(self):
    #     d = NamedFoo(test_data)

    #     self.assertEqual(len(d["a"]),                     6)
    #     self.assertEqual(d["c"],             test_data["c"])
    #     self.assertEqual(d["d"]["e"],   test_data["d"]["e"])
    #     self.assertEqual(d["d"]["f"],   test_data["d"]["f"])
    #     self.assertEqual(d["a/0"],       test_data["a"][0])
    #     self.assertEqual(d["a/1"],       test_data["a"][1])

    #     self.assertListEqual(list(d["a"][2:6]),       [1.0, 2, 3, 4])


if __name__ == '__main__':
    unittest.main()
