import importlib
import pprint
import sys
import unittest


from spdm.Logger import logger
from spdm.data.Entry import Entry


class TestDataEntry(unittest.TestCase):
    def test_new_server(self):
        entry=Entry("mongodb://salmon:2234@127.0.0.1:1234/plasma/1234")
        print(entry)

    def test_new_file(self):
        Entry.file_extents["*.h5"]="hdf5"
        entry=Entry("~/workspace/data/east.h5")
        print(entry)


if __name__ == '__main__':
    unittest.main()
