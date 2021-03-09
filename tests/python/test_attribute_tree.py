import importlib
import pprint
import sys
import unittest


class TestPhysicalGraph(unittest.TestCase):

    def test_attribute_travel(self):

        envs = {
            "name": "WAAA",
            "age": 12
        }

        d = PhysicalGraph({
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
        d, _ = tree_apply_recursive(d, lambda s: s.format_map(envs), str)

        self.assertEqual(d.a[0], "hello world {name}!".format_map(envs))
        self.assertEqual(d.a[1], "hello world2 {name}!".format_map(envs))
        self.assertEqual(d.b.c,  "I'm {age}!".format_map(envs))
        self.assertEqual(d.b.d.e,  "{name} is {age}".format_map(envs))
        self.assertEqual(d.b.d.f,  "{address}")

        self.assertEqual(d.a[2:6],  [1, 2, 3, 4])

    def test_attribute_format(self):
        d = PhysicalGraph({
            'annotation': {'contributors': ['Salmon'],
                           'description': '\\n Just a demo \\n multiline string example\\n',
                           'homepage': 'http://funyun.com/demo.html',
                           'label': 'Genray',
                           'license': 'GPL',
                           'version': '{version}'},
            'build': {'eb': 'Genray/{version}',
                      'toolchain': {'name': 'gompi', 'version': '{FY_TOOLCHAIN_VERSION}'},
                      'toolchainopts': {'pic': True}},

            'run': {'arguments': '-i {equilibrium} -c {config} -n {number_of_steps} -o {OUTPUT}',
                    'exec_cmd': '${EBROOTGENRAY}/bin/xgenray',
                    'module': '{mod_path}/{version}-{tag_suffix}'}}
        )
        envs = {
            "version": '10.8',
            "tag_suffix": "foss-2019",
        }
        format_string_recursive(d,  envs)

        self.assertEqual(d.run.module,  '{mod_path}/10.8-foss-2019')
        self.assertEqual(d.run.arguments,  '-i {equilibrium} -c {config} -n {number_of_steps} -o {OUTPUT}')
        self.assertEqual(d.annotation.version,  '10.8')


if __name__ == '__main__':
    sys.path.append("/home/salmon/workspace/SpDev/SpDB")
    from spdm.data.PhysicalGraph import PhysicalGraph
    from spdm.util.logger import logger
    from spdm.util.dict_util import format_string_recursive, tree_apply_recursive

    unittest.main()
