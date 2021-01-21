import pprint
import unittest
from spdm.util.LazyProxy import LazyProxy


class TestProfile(unittest.TestCase):

    def test_set(self):
        data = {}
        entry = LazyProxy(data)

        entry.a.b.c.d[3] = {"a": [1, 2, 3, 45]}
        entry.a.b.c.e = ["a", "b", "d", "c"]
        entry.a.b.c.e[2] = 2

        res = {'a': {'b': {'c': {'d': {3: {'a': [1, 2, 3, 45]}}, 'e': ['a', 'b', 2, 'c']}}}}
        self.assertEqual(data, res)

    def test_get(self):
        data = {'a': {'b': {'c': {'d': {3: {'a': [1, 2, 3, 45]}}, 'e': ['a', 'b', 2, 'c']}}}}

        entry = LazyProxy(data)

        entry.a.b.c.d[3] = {"a": [1, 2, 3, 45]}
        entry.a.b.c.e = ["a", "b", "d", "c"]
        entry.a.b.c.e[2] = 2
        self.assertEqual(entry.a.b.c.d[3], data["a"]["b"]["c"]["d"][3])
        self.assertEqual(entry.a.b.c.e, ["a", "b", 2, "c"])

    def test_handler(self):

        entry = LazyProxy(None,
                          get=lambda c, o, p: '.'.join(map(str, p)),
                          set=lambda c, o, p, v: None)

        self.assertEqual(entry.a.b.c.__value__(), 'a.b.c')

        self.assertEqual(entry.a.b.c,  'a.b.c')

        entry = LazyProxy(None,
                          get=lambda c, o, p: '.'.join(map(str, p)),
                          get_value=lambda c, o, p: p,
                          set=lambda c,o, p, v: None)

        self.assertEqual(entry.a.b.c.__value__(), ['a', 'b', 'c'])

        self.assertEqual(entry.a.b.c,  'a.b.c')


if __name__ == '__main__':

    unittest.main()
