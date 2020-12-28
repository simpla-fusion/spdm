import importlib
import pprint
import sys
import unittest

from spdm.data.Collection import Collection
from spdm.util.logger import logger


class TestFileCollection(unittest.TestCase):

    def test_file(self):
        collection = Collection("tmp_dir/*.json")
        fp = collection.open_document(12344)
        # logger.debug([p for p in collection.find()])
        # idx = collection.insert_one({"First": "this is a test"})
        # logger.debug(idx)
        # logger.debug(collection.find_one({"_id": idx}))


if __name__ == '__main__':
    unittest.main()
