from logging import log
import unittest

from spdm.common.logger import logger
from spdm.data import List, Dict, Node, sp_property
from spdm.data.Entry import as_entry


class Foo(Dict):
    def __init__(self, data: dict, *args, **kwargs):
        super().__init__(data, *args, **kwargs)

    # @sp_property
    # def a(self) -> float:
    #     return self.get("a")


class Doo(Dict):
    def __init__(self, *args,   **kwargs):
        super().__init__(*args,  **kwargs)

    f0 = sp_property(type_hint=Foo)

    f1: Foo = sp_property()  # recommend

    f2 = sp_property[Foo]()

    @sp_property
    def f3(self) -> Foo:
        return self.get("f3", {})

    # # not support until Python 3.9
    # @sp_property[Foo]
    # def foo1(self)  :
    #     return self.get("foo", {})


if __name__ == '__main__':
    cache = {
        "f0": {"a":  0},
        "f1": {"a":  1},
        "f2": {"a":  2},
        "f3": {"a":  3},

        "foo": {"a": 1234},
        "foo_list": []}

    doo = Doo(cache)

    # logger.debug(doo.f0.dump())
    # logger.debug(doo.f1.dump())
    # logger.debug(doo.f2.dump())
    # logger.debug(doo.f3.dump())

    entry = as_entry(doo)

    logger.debug(type(entry.child("f0").pull()))
    logger.debug(type(entry.child("f1").pull()))
    logger.debug(cache)
