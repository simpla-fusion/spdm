from typing import Generic
import unittest

from spdm.data.Node import Node, Dict, _TObject
from spdm.data.sp_property import sp_property, sp_property_with_parameter
from spdm.util.logger import logger


class Foo(Dict[str, Node]):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def a(self) -> float:
        return self["a"]


class TestXML(unittest.TestCase):
    def test_get(self):
        class D(Dict[str, Node]):
            def __init__(self, *args,   **kwargs):
                super().__init__(*args,  **kwargs)

            @sp_property
            def foo(self) -> Foo:
                return self["foo"]

            @sp_property
            def goo(self) -> None:
                return None
        cache = {"foo": {"a": 1234}}
        d = D(cache)

        logger.debug(d.foo.a)
        logger.debug(cache["foo"].a)


if __name__ == '__main__':
    unittest.main()
