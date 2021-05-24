
import pathlib
import unittest

from spdm.data.File import File
from spdm.util.logger import logger


class TestXML(unittest.TestCase):
    def test_get(self):
        entry = File(pathlib.Path(__file__).parent/"../data/test.xml").entry
        self.assertEqual(entry.get(["timeslice", 0, "eq", "psi"]), "mdsplus://1.2.3.4/east")

    def test_put(self):
        entry = File(pathlib.Path(__file__).parent/"../data/test.xml").entry
        entry.put("hello world!", ["timeslice", 0, "eq", "psi"])
        logger.debug(entry._data)
        self.assertEqual(entry.get(["timeslice", 0, "eq", "psi"]), "hello world!")


if __name__ == '__main__':
    unittest.main()
