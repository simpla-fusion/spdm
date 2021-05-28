import unittest

from spdm.data.Node import Node, Dict, List, _next_, _not_found_
from spdm.util.logger import logger


class TestNode(unittest.TestCase):
    def test_node_initialize(self):
        d = Dict({
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
        d = Dict(cache)

        self.assertEqual(len(d["a"]),                 6)
        self.assertEqual(d["c"],             cache["c"])
        self.assertEqual(d["d"]["e"],   cache["d"]["e"])
        self.assertEqual(d["d"]["f"],   cache["d"]["f"])
        self.assertEqual(d["a"][0],       cache["a"][0])
        self.assertEqual(d["a"][1],       cache["a"][1])
        self.assertEqual(d["a"][2:6],      [1, 2, 3, 4])

        self.assertTrue(d["f"]["g"].empty)

    def test_node_set(self):
        cache = {}

        d = Dict(cache)

        d["a"] = "hello world {name}!"
        d["c"][_next_] = 1.23455
        d["c"][_next_] = {"a": "hello world", "b": 3.141567}

        self.assertEqual(cache["a"], "hello world {name}!")
        self.assertEqual(cache["c"][0],  1.23455)
        self.assertEqual(cache["c"][1]["a"], "hello world")
        self.assertEqual(d["c"][1]["a"], "hello world")

        d["e"]["f"] = 5
        d["e"]["g"] = 6
        self.assertEqual(cache["e"]["f"], 5)
        self.assertEqual(cache["e"]["g"], 6)

    def test_node_insert(self):
        cache = {"this_is_a_cache": True}

        d = Dict(cache)

        d["a"] = "hello world {name}!"
        self.assertEqual(cache["a"], "hello world {name}!")

        d["c"][_next_] = 1.23455
        d["c"][_next_] = {"a": "hello world", "b": 3.141567}

        self.assertEqual(cache["c"][0],  1.23455)

    def test_node_append(self):
        d = List()
        d[_next_] = {"a": 1, "b": 2}

        self.assertEqual(len(d), 1)
        self.assertTrue(d.__category__ | Node.Category.LIST)
        self.assertEqual(d[0]["a"], 1)
        self.assertEqual(d[0]["b"], 2)

    def test_node_boolean(self):
        d = Dict()
        self.assertTrue(d.empty)
        self.assertTrue(d["a"] or 12.3, 12.3)

    def test_node_del(self):
        cache = {
            "a": [
                "hello world {name}!",
                "hello world2 {name}!",
                1, 2, 3, 4
            ]
        }

        d = Node(cache)
        del d["a"]
        self.assertTrue("a" not in cache)

    # def test_decorate(self):

    #     @as_node_tree
    #     class Foo(Mapping[str, Any]):
    #         def __init__(self, d, *args, **kwargs) -> None:
    #             self._cache = d

    #         def __getitem__(self, k):
    #             return self._cache[k]

    #         def __setitem__(self, k, v):
    #             self._cache[k] = v

    #         def __iter__(self) -> Iterator:
    #             yield from self._cache

    #         def __len__(self) -> int:
    #             return len(self._cache)

    #     cache = {}

    #     foo = Foo(cache)

    #     foo.e.f = 5

    #     self.assertEqual(cache["e"]["f"], 5)

    # def test_subclass(self):
    #     class Foo(Node):
    #         @cached_property
    #         def prop_cached(self):
    #             return {"a": 1, "b": 2}

    #         @property
    #         def prop(self):
    #             return {"c": 4, "d": 6}

    #     foo = Foo()

    #     self.assertTrue(foo.prop_cached.a, 1)
    #     self.assertTrue(foo.prop.c, 4)

    # def test_node_format(self):
    #     d = Node({
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
