import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import scipy.stats
from scipy.interpolate import RectBivariateSpline, UnivariateSpline

sys.path.append("/home/salmon/workspace/fytok/python")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")

if __name__ == "__main__":

    from spdm.data.Document import Document
    from spdm.data.Collection import Collection
    from spdm.data.File import File
    from spdm.util.logger import logger

    entry = Collection("EAST+mdsplus:///home/salmon/public_data/~t/?tree_name=efit_east")\
        .open(shot=55555, time_slice=20, mode="r").entry
    # entry = Document({"path": ["/home/salmon/workspace/fytok/devices/EAST/imas/3/dynamic/config.xml",
    #                            "/home/salmon/workspace/fytok/devices/EAST/imas/3/static/config.xml"
    #                            ],
    #                   "schema": "file/XML"}).entry
    # doc = File({"path": ["/home/salmon/workspace/fytok/devices/EAST/imas/3/dynamic/config.xml",
    #                      "/home/salmon/workspace/fytok/devices/EAST/imas/3/static/config.xml"
    #                      ],
    #             "file_format": "XML"})
    for coil in entry.pf_active.coil:
        logger.debug(coil.current.__value__())
