import os
import pathlib
import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
from spdm import open_entry, open_db
from spdm.util.logger import logger
from spdm.data.Collection import Collection
os.environ["SP_DATA_MAPPING_PATH"] = "/home/salmon/workspace/fytok_data/mapping"

if __name__ == '__main__':

    # db = open_db("localdb+hdf5:///home/salmon/workspace/output/local_db/{shot}_{run}", mode="rw")
    entry = open_entry("mdsplus[EAST]://202.127.204.12?tree_name=efit_east#38300")
    # entry = open_entry("file+mdsplus[EAST]:///home/salmon/workspace/data/~t/?tree_name=efit_east#38300")
    logger.debug(entry.get(["equilibrium","time_slice",100]).dump())

    del entry
    # db.insert_one(
    #     {"equilibrium": entry.get(["equilibrium"]).dump()},
    #     shot=123, run=2
    # )