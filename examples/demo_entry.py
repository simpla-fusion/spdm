import os
import pathlib

from spdm.data.Entry import open_entry
from spdm.utils.logger import logger

os.environ["SP_DATA_MAPPING_PATH"] = "/home/salmon/workspace/fytok_data/mapping"


if __name__ == '__main__':

    DATA_PATH = pathlib.Path("/home/salmon/workspace/fytok_data/gfiles")

    eq0 = open_entry(f"file+geqdsk:///{DATA_PATH.as_posix()}/g063982.04800", mode="r").fetch()

    eq1 = open_entry(DATA_PATH/"g063982.04800", mode="r", format="geqdsk").fetch()

    logger.debug(eq0)

    logger.debug(eq1)

    eq2 = open_entry(f"east+mdsplus:///home/salmon/workspace/fytok_data/mdsplus/~t/", shot=70745)

    logger.debug(eq2.child("equilibrium/time_slice/0/boundary/outline/r").fetch())

    # # shot_num = 70754

    # time_slice = 10

    # entry = open_entry(f"east://202.127.204.12#{shot_num}")

    # eq = entry.child(f"equilibrium/time_slice/{time_slice}/").fetch()

    # with File(f"./g{shot_num}", mode="w", format="geqdsk") as fid:
    #     fid.write(eq)
