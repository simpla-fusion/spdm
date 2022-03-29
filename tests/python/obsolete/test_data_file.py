import unittest

from spdm.data.File import File
from spdm.logger import logger


class TestFile(unittest.TestCase):
    def test_geqdsk(self):
        gfile = File(
            # "/home/salmon/workspace/fytok/examples/data/NF-076026/geqdsk_550s_partbench_case1",
            "/home/salmon/workspace/data/15MA inductive - burn/Standard domain R-Z/High resolution - 257x513/g900003.00230_ITER_15MA_eqdsk16HR.txt",
            # "/home/salmon/workspace/data/Limiter plasmas-7.5MA li=1.1/Limiter plasmas 7.5MA-EQDSK/Limiter_7.5MA_outbord.EQDSK",
            format="geqdsk")

        self.assertEqual(gfile.entry.get("vacuum_toroidal_field.r0"), 6.2)

    def test_xml(self):
        device = File("/home/salmon/workspace/fytok/data/mapping/ITER/imas/3/static/config.xml")
        retcangle = device.entry.moveto(["pf_active.coil", 0, "element.geometry.rectangle"]).pull(lazy=False)

        self.assertEqual(retcangle["height"], 2.12)

        wall_r = device.entry.moveto(["wall.description_2d", 0, "vessel.annular.outline_inner.r"]).pull(lazy=False)
        logger.debug(wall_r)


if __name__ == '__main__':
    unittest.main()
