import pprint
import unittest

from spdm.util.entry import ContainerEntry


class TestPointer(unittest.TestCase):

    def test_read(self):
        d = {"a": {"b": 1234}, "c": 1234}
        p = ContainerEntry(d).root
        self.assertEqual(p.actor.physics.bout._prefix,
                         ("actor", "physics", "bout"))
        self.assertEqual(p.a.b, d["a"]["b"])
        self.assertEqual(p["a/b"], d["a"]["b"])
        print(p.a.b[4:5, 6:10, 5])
        p.ab.c.d = 1234
        pprint.pprint(d)

    def test_write(self):
        d = {}
        root = ContainerEntry(d).root
        root.a.b = 123.4
        self.assertEqual(123.4, d["a"]["b"])


if __name__ == '__main__':
    unittest.main()
