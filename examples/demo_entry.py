import os

from spdm.data.open_entry import File, open_entry
from spdm.utils.logger import logger

os.environ["SP_DATA_MAPPING_PATH"] = "/home/salmon/workspace/fytok_data/mapping"

if __name__ == '__main__':

    # db: Collection = open_db("mdsplus[EAST]://202.127.204.12")

    entry = open_entry("MDSplus[EAST]://202.127.204.12?tree_name=pcs_east#70754")

    ip = entry.child(("tf", "coil",  "current", "data")).query()
    pf = entry.child("pf_active").query()

    logger.debug(ip)
    logger.debug(pf)

    # entry2 = open_entry("file+mdsplus[EAST]:///home/salmon/workspace/data/~t/?tree_name=efit_east#38300")
    # logger.debug(entry2.get(["magnetics"]).dump())

    shot_num = 70754

    time_slice = 10

    entry = open_entry(f"MDSplus[EAST]://202.127.204.12?tree_name=east_efit#{shot_num}")

    eq = entry.child(f"equilibrium/time_slice/{time_slice}").query()

    with File(f"./g{shot_num}", mode="w", format="geqdsk") as fid:
        fid.write(eq)
