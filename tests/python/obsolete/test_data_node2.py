import unittest

from spdm.data  import Node, Dict, List, _next_, _not_found_, sp_property
from spdm.util.logger import logger
from copy import copy, deepcopy


class TestNode2(unittest.TestCase):
    data = {
        "a": [
            "hello world {name}!",
            "hello world2 {name}!",
            1, 2, 3, 4
        ],
        "c": "I'm {age}!",
        "d": {
            "e": "{name} is {age}",
            "f": "{address}"
        }
    }

    def test_dict_initialize(self):
        class Poo:
            def __init__(self, a) -> None:
                print(a)
                self._value = a

        class Foo(Dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            @sp_property
            def a(self) -> Poo:
                return self.get("a")


if __name__ == '__main__':
    unittest.main()
