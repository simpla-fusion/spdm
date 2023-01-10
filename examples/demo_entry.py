import os
import pathlib
import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
from spdm import open_entry, open_db, File
from spdm.util.logger import logger
from spdm.data.Collection import Collection
os.environ["SP_DATA_MAPPING_PATH"] = "/home/salmon/workspace/fytok_data/mapping"

if __name__ == '__main__':

    # db: Collection = open_db("mdsplus[EAST]://202.127.204.12")

    entry = open_entry("mdsplus[EAST]://202.127.204.12?tree_name=pcs_east#70754")

    ip = entry.get(["tf", "coil",  "current", "data"])
    pf = entry.get(["pf_active"]).dump()

    logger.debug(ip)
    logger.debug(pf)

    # entry2 = open_entry("file+mdsplus[EAST]:///home/salmon/workspace/data/~t/?tree_name=efit_east#38300")
    # logger.debug(entry2.get(["magnetics"]).dump())

    shot_num = 70754
    time_slice = 10
    entry = open_entry(f"mdsplus[EAST]://202.127.204.12?tree_name=east_efit#{shot_num}")
    eq = entry.get(["equilibrium", "time_slice", time_slice]).dump()

    with File(f"./g{shot_num}", mode="w", format="geqdsk") as fid:
        fid.write(eq)
