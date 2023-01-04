import os
import pathlib
import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
from spdm.util.logger import logger
from spdm.data.Connection import Connection
from spdm.data.open_entry import open_db


os.environ["SP_DATA_MAPPING_PATH"] = "/home/salmon/workspace/fytok_data/mapping"

if __name__ == '__main__':

    # db = open_db("mdsplus[EAST]:///home/salmon/workspace/data/~t/?tree_name=efit_east")

    # entry = db.find_one(38300)

    # logger.debug(entry.get(["pf_active"]).dump())
    
    db = open_db("mdsplus[EAST]://202.127.204.12?tree_name=efit_east")

    entry = db.find_one(114730)

    logger.debug(entry.get(["pf_active"]).dump())