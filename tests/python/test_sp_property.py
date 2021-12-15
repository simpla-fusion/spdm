import unittest

from spdm.common.logger import logger

from spdm.data import Dict, List, Node, Link, Path, Query, sp_property


class Foo(Dict):
    def __init__(self, data: dict, *args, **kwargs):
        super().__init__(data, *args, **kwargs)

    @sp_property
    def a(self) -> float:
        return self.get("a")


class Doo(Dict):
    def __init__(self, *args,   **kwargs):
        super().__init__(*args,  **kwargs)

    @sp_property
    def foo(self) -> Foo:
        return self.get("foo", {})

    @sp_property
    def goo(self) -> Foo:
        return {"a": 3.14}

    @sp_property
    def foo_list(self) -> List[Foo]:
        return self.get("foo_list", [])

    balaaa = sp_property[Foo](default={"bala": 1})


class TestSpProperty(unittest.TestCase):
    def test_get(self):
        cache = {"foo": {"a": 1234}, }
        d = Doo(cache)

        self.assertFalse(isinstance(cache["foo"], Foo))
        self.assertTrue(isinstance(d.foo, Foo))
        self.assertTrue(isinstance(cache["foo"], Foo))

        self.assertTrue(isinstance(d.balaaa, Foo))
        self.assertTrue(isinstance(cache["balaaa"], Foo))

        self.assertEqual(d.foo.a, cache["foo"].a)

        self.assertEqual(d.goo.a, 3.14)
        self.assertEqual(cache["goo"].a, 3.14)

    def test_get_list(self):

        cache = {"foo_list": [{"a": 1234}, {"b": 1234}, {"c": 1234}, ]}

        d = Doo(cache)

        self.assertFalse(isinstance(cache["foo_list"], Foo))
        self.assertTrue(isinstance(d.foo_list, List))
        self.assertTrue(isinstance(cache["foo_list"], List))
        self.assertTrue(isinstance(d.foo_list[0], Foo))

        self.assertEqual(d.foo_list[0]["a"], 1234)

    def test_set(self):
        cache = {"foo": {"a": 1234}}
        d = Doo(cache)
        self.assertEqual(cache["foo"]["a"], 1234)
        d.foo.a = 45678
        self.assertEqual(cache["foo"].a, 45678)

    def test_delete(self):
        cache = {"foo": {"a": 1234}}
        d = Doo(cache)
        self.assertEqual(cache["foo"]["a"], 1234)
        del d.foo
        self.assertTrue("foo" not in cache)


if __name__ == '__main__':
    unittest.main()
