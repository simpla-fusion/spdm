import unittest
from copy import deepcopy

from spdm.data.Path import Path
from spdm.utils.logger import logger
from spdm.utils.tags import _not_found_
from spdm.utils.typing import get_origin, get_args
import typing


class TestTyping(unittest.TestCase):

    def test_get_origin(self):
        _T = typing.TypeVar("_T")

        class Foo(typing.Generic[_T]):
            pass

        foo = Foo[int]()

        self.assertEqual(get_origin(int), int)
        self.assertEqual(get_origin(Foo), Foo)
        self.assertEqual(get_origin(Foo[int]), Foo)

        class Goo(Foo[_T]):
            pass

        self.assertEqual(get_origin(Goo), Goo)
        self.assertEqual(get_origin(Goo[int]), Goo)

        class Doo(Foo[float]):
            pass

        self.assertEqual(get_origin(Doo), Doo)

        self.assertEqual(get_origin(1.234), float)
        self.assertEqual(get_origin(Foo()), Foo)
        self.assertEqual(get_origin(Foo[int]()), Foo)
        self.assertEqual(get_origin(Goo[int]()), Goo)
        self.assertEqual(get_origin(Doo()), Doo)

    def test_get_args(self):
        _T = typing.TypeVar("_T")

        class Foo(typing.Generic[_T]):
            pass

        self.assertEqual(get_args(int), tuple())
        self.assertEqual(get_args(Foo), tuple())
        self.assertEqual(get_args(Foo[int]), (int,))

        class Goo(Foo[_T]):
            pass

        self.assertEqual(get_args(Goo), tuple())
        self.assertEqual(get_args(Goo[int])[0], int)

        class Doo(Foo[float]):
            pass

        self.assertEqual(get_args(Doo), (float,))


if __name__ == '__main__':
    unittest.main()
