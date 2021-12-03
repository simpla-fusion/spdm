import unittest

from spdm.common.logger import logger
from spdm.data.Dict import Dict
from spdm.data.List import List
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property


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


class TestSpProperty(unittest.TestCase):
    def test_get(self):
        cache = {"foo": {"a": 1234}}
        d = Doo(cache)

        self.assertFalse(isinstance(cache["foo"], Foo))
        self.assertTrue(isinstance(d.foo, Foo))
        self.assertTrue(isinstance(cache["foo"], Foo))

        self.assertEqual(d.foo.a, cache["foo"].a)
        d.goo.a
        self.assertEqual(cache["goo"].a, 3.14)

        logger.debug(cache)

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
