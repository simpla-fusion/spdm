
from functools import cached_property
import pathlib
from typing import Generic
import unittest

from spdm.data.Node import Node, Dict, _TObject
from spdm.data.sp_property import sp_property
from spdm.util.logger import logger


class Foo(Dict[str, Node]):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def a(self) -> float:
        return self["a"]


def sp_property_(prop: Generic[_TObject]) -> sp_property[_TObject]:
    return sp_property[prop]


class TestXML(unittest.TestCase):
    def test_get(self):
        class D(Dict[str, Node]):
            def __init__(self, *args,   **kwargs):
                super().__init__(*args,  **kwargs)

            @sp_property[Foo]
            def foo(self) -> Foo:
                return Foo(self["foo"])

        d = D({"foo": {"a": 1234}})
        logger.debug(d.foo.a)


if __name__ == '__main__':
    unittest.main()
