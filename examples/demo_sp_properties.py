import unittest

from spdm.common.logger import logger
from spdm.data.Dict import Dict
from spdm.data.List import List
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property


class Foo(Dict):
    def __init__(self, data: dict, *args, **kwargs):
        super().__init__(data, *args, **kwargs)

    # @sp_property
    # def a(self) -> float:
    #     return self.get("a")


class Doo(Dict):
    def __init__(self, *args,   **kwargs):
        super().__init__(*args,  **kwargs)

    # @sp_property
    # def foo(self) -> Foo:
    #     return self.get("foo", {})

    # @sp_property
    # def goo(self) -> Foo:
    #     return {"a": 3.14}

    # @sp_property
    # def foo_list(self) -> List[Foo]:
    #     return self.get("foo_list", [])

    foo = sp_property[Foo]()

    @sp_property
    def foo1(self) -> Foo:
        return self.get("foo", {})

    # # not support until Python 3.9
    # @sp_property[Foo]
    # def foo1(self)  :
    #     return self.get("foo", {})


if __name__ == '__main__':
    cache = {"foo": {"a": 1234}, "foo_list": []}

    doo = Doo(cache)

    logger.debug(doo.foo)
