import unittest
from logging import log

from dataclasses import dataclass
from spdm.data.HTree import HTree
from spdm.data.sp_property import sp_property, sp_tree
from spdm.utils.logger import logger


@sp_tree
class Foo:
    first: float
    second: str = "b"

    def goo(self):
        print("goo")


if __name__ == '__main__':

    doo = Foo(a=1.0, b="b")
    logger.debug(Foo)
    logger.info(doo.first)
    logger.info(doo.second)
