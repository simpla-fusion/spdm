import unittest
from copy import deepcopy
from logging import log

from spdm.util.logger import logger
from spdm.common.tags import _not_found_
from spdm.data.Path import Path


class TestPath(unittest.TestCase):

    def test_tuple(self):
        p = ['a', 'b', {'c':  slice(4, 10, 2)}]
        self.assertEqual(Path(*p), p)

    def test_str(self):
        p_str = 'a/b/c[4:10:-2]/d/hello/[1,2,a]/{c,d,e}'
        p = ['a', 'b', 'c', slice(4, 10, -2), 'd', 'hello', [1, 2, 'a'], {'c', 'd', 'e'}]
        self.assertEqual(Path(p_str)[:], p)
        self.assertEqual(str(Path(p)), p_str)

    def test_compare(self):
        p_str = 'a/b/c[4:10:-2]/d/hello'
        p = ('a', 'b', 'c',  slice(4, 10, -2), 'd', 'hello')
        self.assertEqual(Path(p_str), Path(p))

    def test_parent(self):
        p = ('a', 'b', 'c', slice(4, 10, -2), 'd', 'hello')
        self.assertEqual(Path(p).parent, Path(p[:-1]))

    def test_query(self):

        p = Path('a/b/c', slice(4, 10, 2), 'd/hello', {"$eq": 1}, "a/b/c/d")

        query, predicate = p.as_query()

        logger.debug(query)


if __name__ == '__main__':
    unittest.main()
