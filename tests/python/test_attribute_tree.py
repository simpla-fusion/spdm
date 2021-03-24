import importlib
import pprint
import sys
import unittest
from spdm.util.logger import logger
from spdm.data.Entry import _next_
from spdm.data.Node import Node
from spdm.data.AttributeTree import AttributeTree


class TestAttributeTree(unittest.TestCase):
    def test_node_initialize(self):
        d = Node({
            "c": "I'm {age}!",
            "d": {
                "e": "{name} is {age}",
                "f": "{address}"
            }
        })

    def test_node_get(self):
        cache = {
            "a": [
                "hello world {name}!",
                "hello world2 {name}!",
                1, 2, 3, 4
            ],
            "c": "I'm {age}!",
            "d": {
                "e": "{name} is {age}",
                "f": "{address}"
            }
        }
        d = Node(cache)

        self.assertEqual(len(d["a"]),  6)
        self.assertEqual(d["c"],  cache["c"])
        self.assertEqual(d["d.e"], cache["d"]["e"])
        self.assertEqual(d["d.f"],  cache["d"]["f"])
        self.assertEqual(d["a"][0], cache["a"][0])
        self.assertEqual(d["a"][1],  cache["a"][1])

        # self.assertEqual(d["c"],  "I'm {age}!")
        # self.assertEqual(d["d.e"],  "{name} is {age}")
        # self.assertEqual(d["d.f"],  "{address}")
        # self.assertEqual(d["a"][0], "hello world {name}!")
        # self.assertEqual(d["a"][1], "hello world2 {name}!")

    def test_node_insert(self):
        cache = {}

        d = Node(cache)

        d["a"] = "hello world {name}!"
        self.assertEqual(cache["a"], "hello world {name}!")

        d["c"][_next_] = 1.23455
        d["c"][_next_] = {"a": "hello world", "b": 3.141567}

        self.assertEqual(cache["c"][0],  1.23455)

    def test_node_iter(self):
        d = Node([1, 2, 3, 4, 5, 6])
        self.assertEqual([v for v in d],  [1, 2, 3, 4, 5, 6])

    # def test_attribute_get(self):

    #     d = AttributeTree({
    #         "a": [
    #             "hello world {name}!",
    #             "hello world2 {name}!",
    #             1, 2, 3, 4
    #         ],
    #         "b": {
    #             "c": "I'm {age}!",
    #             "d": {
    #                 "e": "{name} is {age}",
    #                 "f": "{address}"
    #             }
    #         }
    #     })

    #     self.assertEqual(d.a[0], "hello world {name}!")
    #     self.assertEqual(d.a[1], "hello world2 {name}!")
    #     self.assertEqual(d.b.c,  "I'm {age}!")
    #     self.assertEqual(d.b.d.e,  "{name} is {age}")
    #     self.assertEqual(d.b.d.f,  "{address}")
        # self.assertEqual(d.a[2:6], [1, 2, 3, 4])

    # def test_attribute_format(self):
    #     d = PhysicalGraph({
    #         'annotation': {'contributors': ['Salmon'],
    #                        'description': '\\n Just a demo \\n multiline string example\\n',
    #                        'homepage': 'http://funyun.com/demo.html',
    #                        'label': 'Genray',
    #                        'license': 'GPL',
    #                        'version': '{version}'},
    #         'build': {'eb': 'Genray/{version}',
    #                   'toolchain': {'name': 'gompi', 'version': '{FY_TOOLCHAIN_VERSION}'},
    #                   'toolchainopts': {'pic': True}},

    #         'run': {'arguments': '-i {equilibrium} -c {config} -n {number_of_steps} -o {OUTPUT}',
    #                 'exec_cmd': '${EBROOTGENRAY}/bin/xgenray',
    #                 'module': '{mod_path}/{version}-{tag_suffix}'}}
    #     )
    #     envs = {
    #         "version": '10.8',
    #         "tag_suffix": "foss-2019",
    #     }
    #     format_string_recursive(d,  envs)

    #     self.assertEqual(d.run.module,  '{mod_path}/10.8-foss-2019')
    #     self.assertEqual(d.run.arguments,  '-i {equilibrium} -c {config} -n {number_of_steps} -o {OUTPUT}')
    #     self.assertEqual(d.annotation.version,  '10.8')


if __name__ == '__main__':
    unittest.main()
