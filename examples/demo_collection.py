import os
import pathlib
import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
from spdm.util.logger import logger
from spdm.data.Connection import Connection
from spdm.data import open_collection

from spdm import open_entry

if __name__ == '__main__':

    db = open_collection("mdsplus[EAST]:///home/salmon/workspace/data/~t/?tree_name=efit_east",
                         mapping="/home/salmon/workspace/fytok_data/mapping")

    entry = db.find(38300)

    logger.debug(entry.get(["pf_active"]))
