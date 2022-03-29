import importlib
import pprint
import sys
import unittest
from spdm.data.Collection import Collection
from spdm.logger import logger


class TestFileCollection(unittest.TestCase):

    def test_file(self):
        # collection = Collection("tmp_dir/{id:06}.json", mode="x")
        collection = Collection("file:///home/salmon/workspace/output/tmp_dir/*.json", mode="x")
        # collection = Collection("geqdsk://~/tmp_dir/*")

        fp = collection.create(12344)
        fp.write({"a": 1234.5})

        # logger.debug([p for p in collection.find()])
        # idx = collection.insert_one({"First": "this is a test"})
        # logger.debug(idx)
        # logger.debug(collection.find_one({"_id": idx}))


if __name__ == '__main__':

    unittest.main()
