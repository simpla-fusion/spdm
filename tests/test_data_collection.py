import importlib
import pprint
import sys
import unittest

from spdm.data import io
from spdm.util.logger import logger


class TestFileCollection(unittest.TestCase):

    def test_file(self):
        collection = io.connect("local+json://tmp_dir")
        logger.debug([p for p in collection.find()])
        idx = collection.insert_one({"First": "this is a test"})
        logger.debug(idx)
        logger.debug(collection.find_one({"_id": idx}))


if __name__ == '__main__':
    unittest.main()
