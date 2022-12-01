import os
import pathlib
import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
from ..util.logger import logger
from spdm.data.File import File
from spdm.data.Mapping import Mapping

from spdm import open_entry

if __name__ == '__main__':

    # entry = Mapping("/home/salmon/workspace/fytok_data/mapping")\
    #     .map(File("/home/salmon/workspace/data/~t/?tree_name=efit_east,shot=38300",
    #               format="mdsplus").read(), source_schema="EAST")
    entry = open_entry("mdsplus[EAST]:///home/salmon/workspace/data/~t/?tree_name=efit_east#38300",
                       mapping_path="/home/salmon/workspace/fytok_data/mapping")

    logger.debug(entry.get(["pf_active"]))
