import unittest

from spdm.data.Node import Dict, _TObject, sp_property
from spdm.util.logger import logger


class Foo(Dict[_TObject]):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def a(self) -> float:
        return self["a"]


class Doo(Dict):
    def __init__(self, *args,   **kwargs):
        super().__init__(*args,  **kwargs)

    @sp_property
    def foo(self) -> Foo[str]:
        return self["foo"]

    @sp_property
    def goo(self) -> None:
        return None


class TestSpProperty(unittest.TestCase):
    def test_get(self):
        cache = {"foo": {"a": 1234}}
        d = Doo(cache)

        self.assertFalse(isinstance(cache["foo"], Foo))
        self.assertTrue(isinstance(d.foo, Foo))
        self.assertTrue(isinstance(cache["foo"], Foo))
        self.assertTrue(cache["foo"].__orig_class__== Foo[str])

        self.assertEqual(d.foo.a, cache["foo"]["a"])

    def test_insert(self):
        cache = {"foo": {"a": 1234}}
        d = Doo(cache)
        self.assertEqual(cache["foo"]["a"], 1234)
        d.foo.a = 45678
        self.assertEqual(cache["foo"]["a"], 45678)


if __name__ == '__main__':
    unittest.main()
