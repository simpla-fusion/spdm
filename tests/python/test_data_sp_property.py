import unittest
from typing import Generic

from spdm.data.Node import Dict, Node, _TObject, sp_property
from spdm.util.logger import logger


class Foo(Dict[Node]):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    @sp_property
    def a(self) -> float:
        return self["a"]


class Doo(Dict[Node]):
    def __init__(self, *args,   **kwargs):
        super().__init__(*args,  **kwargs)

    @sp_property
    def foo(self) -> Foo:
        return self["foo"]

    @sp_property
    def goo(self) -> None:
        return None


class TestXML(unittest.TestCase):
    def test_get(self):
        cache = {"foo": {"a": 1234}}
        d = Doo(cache)
        self.assertTrue(isinstance(cache["foo"], Foo))
        self.assertEqual(d.foo.a, cache["foo"]["a"])

    def test_put(self):
        cache = {"foo": {"a": 1234}}
        d = Doo(cache)
        d.foo.a = 45678
        self.assertEqual(cache["foo"]["a"], 45678)


if __name__ == '__main__':
    unittest.main()
