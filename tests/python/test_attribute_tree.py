import importlib
import pprint
import sys
import unittest


class TestAttributeTree(unittest.TestCase):

    def test_attribute_travel(self):

        envs = {
            "name": "WAAA",
            "age": 12
        }
        d = AttributeTree({
            "a": [
                "hello world {name}!",
                "hello world2 {name}!",
                1, 2, 3, 4
            ],
            "b": {
                "c": "I'm {age}!",
                "d": {
                    "e": "{name} is {age}",
                    "f": "{address}"
                }
            }
        })

        d.__apply_recursive__(lambda s: s.format_map(envs), str)

        self.assertEqual(d.a[0], "hello world {name}!".format_map(envs))
        self.assertEqual(d.a[1], "hello world2 {name}!".format_map(envs))
        self.assertEqual(d.b.c,  "I'm {age}!".format_map(envs))
        self.assertEqual(d.b.d.e,  "{name} is {age}".format_map(envs))
        self.assertEqual(d.b.d.f,  "{address}")

        self.assertEqual(d.a[2:6],  [1, 2, 3, 4])


if __name__ == '__main__':
    sys.path.append("/home/salmon/workspace/SpDev/SpDB")
    from spdm.util.AttributeTree import AttributeTree
    from spdm.util.logger import logger

    unittest.main()
