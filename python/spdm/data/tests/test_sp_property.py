import unittest

from spdm.utils.logger import logger
from spdm.data.List import List
from spdm.data.sp_property import SpPropertyClass, sp_property


class Foo(SpPropertyClass):
    a: float = sp_property()


class Goo(SpPropertyClass):
    def __init__(self, *args,   **kwargs) -> None:
        super().__init__(*args,   **kwargs)
    b: float = sp_property()


class Doo(SpPropertyClass):

    foo: Foo = sp_property(default_value={"a": 1})

    goo: Goo = sp_property(default_value={"b": 3.14})

    foo_list: List[Foo] = sp_property()

    balaaa = sp_property[Foo](default={"bala": 1})


class TestSpProperty(unittest.TestCase):
    def test_get(self):
        cache = {"foo": {"a": 1234}, }
        d = Doo(cache=cache)

        self.assertFalse(isinstance(cache["foo"], Foo))
        self.assertTrue(isinstance(d.foo, Foo))
        self.assertTrue(isinstance(cache["foo"], Foo))

        self.assertTrue(isinstance(d.balaaa, Foo))
        self.assertTrue(isinstance(cache["balaaa"], Foo))

        self.assertEqual(d.foo.a, cache["foo"].a)

        self.assertEqual(d.goo.b, 3.14)
        self.assertEqual(cache["goo"].b, 3.14)

    def test_get_list(self):

        cache = {"foo_list": [{"a": 1234}, {"b": 1234}, {"c": 1234}, ]}

        d = Doo(cache=cache)

        self.assertFalse(isinstance(cache["foo_list"], Foo))
        self.assertTrue(isinstance(d.foo_list, List))
        self.assertTrue(isinstance(cache["foo_list"], List))
        self.assertTrue(isinstance(d.foo_list[0], Foo))

        self.assertEqual(d.foo_list[0]["a"], 1234)

    def test_set(self):
        cache = {"foo": {"a": 1234}}
        d = Doo(cache=cache)
        self.assertEqual(cache["foo"]["a"], 1234)
        d.foo.a = 45678
        self.assertEqual(cache["foo"].a, 45678)

    def test_delete(self):
        cache = {"foo": {"a": 1234}}
        d = Doo(cache=cache)
        self.assertEqual(cache["foo"]["a"], 1234)
        del d.foo
        self.assertTrue("foo" not in cache)


if __name__ == '__main__':
    unittest.main()
