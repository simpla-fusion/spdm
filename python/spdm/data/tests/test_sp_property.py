import unittest

from spdm.utils.logger import logger
from spdm.data.List import List, AoS
from spdm.data.sp_property import SpDict, sp_property


class Foo(SpDict):
    a: float = sp_property(default_value=4)
    b: float = sp_property()
    c: float = sp_property()


class Goo(SpDict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    value: float = sp_property()

    foos: List[Foo] = sp_property(default_value={"a": 1, "b": 2, "c": 3})


class Doo(SpDict):

    foo: Foo = sp_property(default_value={"a": 1})

    goo: Goo = sp_property(default_value={"value": 3.14})

    foo_list: List[Foo] = sp_property()

    balaaa = sp_property[Foo](default={"bala": 1})


class TestSpProperty(unittest.TestCase):
    def test_get(self):
        cache = {"foo": {"a": 1234}, }
        d = Doo(cache)

        self.assertFalse(isinstance(cache["foo"], Foo))
        self.assertTrue(isinstance(d.foo, Foo))
        self.assertTrue(isinstance(d._cache["foo"], Foo))

        self.assertTrue(isinstance(d.balaaa, Foo))
        self.assertTrue(isinstance(d._cache["balaaa"], Foo))

        self.assertEqual(d.foo.a, d._cache["foo"].a)

    def test_default_value(self):

        d = Doo()
        self.assertEqual(d.goo.value, 3.14)
        self.assertEqual(d._cache["goo"].value, 3.14)

    def test_get_list(self):

        cache = {"foo_list": [{"a": 1234}, {"b": 1234}, {"c": 1234}, ]}

        d = Doo(cache)

        self.assertFalse(isinstance(cache["foo_list"], Foo))
        self.assertTrue(isinstance(d.foo_list, List))
        # self.assertTrue(isinstance(cache["foo_list"], List))
        self.assertTrue(isinstance(d.foo_list[0], Foo))

        self.assertEqual(d.foo_list[0]["a"].__value__, 1234)

    def test_set(self):
        cache = {"foo": {"a": 1234}}
        d = Doo(cache)
        self.assertEqual(cache["foo"]["a"], 1234)
        d.foo.a = 45678.0
        self.assertEqual(d._cache["foo"].a, 45678)

    def test_delete(self):
        cache = {"foo": {"a": 1234}}
        d = Doo(cache)
        self.assertEqual(cache["foo"]["a"], 1234)
        del d.foo
        self.assertTrue("foo" not in cache)

    def test_list_default_child_value(self):
        cache = [{"a": 6}, {"a": 7}, {"a": 8}, {}]
        d = AoS[Foo](cache, default_value={"a": 1, "b": 2, "c": 3})
        self.assertEqual(d[0].a, 6)
        self.assertEqual(d[0].b, 2)
        self.assertEqual(d[0].c, 3)
        self.assertEqual(d[-1].a, 4)
        self.assertEqual(d[-1].b, 2)
        self.assertEqual(d[-1].c, 3)


if __name__ == '__main__':
    unittest.main()
