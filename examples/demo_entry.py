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

    # db: Collection = open_db("mdsplus[EAST]://202.127.204.12")
  
    entry = open_entry("mdsplus[EAST]://202.127.204.12#70754")
    

    logger.debug(entry.get(["tf"]).dump())

    # entry2 = open_entry("file+mdsplus[EAST]:///home/salmon/workspace/data/~t/?tree_name=efit_east#38300")
    # logger.debug(entry2.get(["magnetics"]).dump())