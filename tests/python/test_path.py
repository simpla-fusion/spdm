import unittest
from copy import deepcopy
from logging import log

from spdm.util.logger import logger
from spdm.common.tags import _not_found_
from spdm.data.Path import Path


from spdm.data import Path


class TestPath(unittest.TestCase):

    def test_tuple(self):
        p = ('a', 'b', 'c', 2, slice(4, 10, -2))
        self.assertEqual(Path(*p).items, p)

    def test_str(self):
        p_str = 'a/b/c/2/4:10:-2/d/hello/[1,2,a]/{c,d,e}'
        p = ('a', 'b', 'c', 2, slice(4, 10, -2), 'd', 'hello', [1, 2, 'a'], {'c', 'd', 'e'})
        self.assertEqual(Path(p_str).items, p)
        self.assertEqual(Path(str(Path(p))), Path(p))
        logger.debug(Path(p_str).items)

    def test_compare(self):
        p_str = 'a/b/c/2/4:10:-2/d/hello'
        p = ('a', 'b', 'c', 2, slice(4, 10, -2), 'd', 'hello')
        self.assertEqual(Path(p_str), Path(p))

    def test_parent(self):
        p = ('a', 'b', 'c', 2, slice(4, 10, -2), 'd', 'hello')
        self.assertEqual(Path(p).parent, Path(p[:-1]))

        logger.debug(Path(p))


if __name__ == '__main__':
    unittest.main()
