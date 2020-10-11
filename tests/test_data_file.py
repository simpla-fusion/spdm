import json
import pprint
import unittest
import pathlib

from spdm.data import bag
from spdm.util.logger import logger


class TestFlowBag(unittest.TestCase):

    dict_data = {"glossary": {
        "title": "example glossary",
        "GlossDiv": {
            "title": "S",
            "GlossList": {
                "GlossEntry": {
                    "ID": "SGML",
                    "SortAs": "SGML",
                    "GlossTerm": "Standard Generalized Markup Language",
                    "Acronym": "SGML",
                    "Abbrev": "ISO 8879:1986",
                    "GlossDef": {
                        "para": "A meta-markup language, used to create markup languages such as DocBook.",
                        "GlossSeeAlso": ["GML", "XML"]
                    },
                    "GlossSee": "markup"
                }
            }
        }
    }
    }

    def testFileRW(self):
        fn = "~/test.json"
        fp = pathlib.Path(fn).expanduser()

        file_type = "json"

        bag.File(fn, schema={"file_type": file_type})\
            .write(TestFlowBag.dict_data)

        d2 = json.load(open(fp))
        self.assertEqual(TestFlowBag.dict_data, d2)
        d3 = bag.File(fn, schema={"file_type": file_type}).read()
        self.assertEqual(TestFlowBag.dict_data, d3)


if __name__ == '__main__':
    unittest.main()
