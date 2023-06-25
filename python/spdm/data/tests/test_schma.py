import unittest
from copy import deepcopy

from spdm.utils.tags import _not_found_
from spdm.data.Schema import Schema
from spdm.data.File import File
import xmlschema

from jsonschema import validate

EQ_PATH = "/home/salmon/workspace/data/15MA inductive - burn/Increased domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16VVHR.txt"
DD_PATH = "/home/salmon/workspace/data-dictionary/dd_physics_data_dictionary.xsd"


class TestSchema(unittest.TestCase):

    def test_valid(self):

        xsd_schema = xmlschema.XMLSchema("/home/salmon/workspace/SpDB/python/spdm/data/tests/test_schema.xsd")
        xsd_schema.to_dict()

        # person = {"name": "John Doe", "age": 30}

        # self.assertTrue(xsd_schema.is_valid(person))

    # def test_valid_ids(self):
    #     eq = File(EQ_PATH, format="GEQdsk").read().dump()

    def test_valid_json(self):
        
        from jsonschema import validate
        # Define the schema
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }

        # Define the Python object
        data = {"name": "John", "age": 30}

        # Validate the object against the schema
        validate(instance=data, schema=schema)


if __name__ == '__main__':
    unittest.main()
