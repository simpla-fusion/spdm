import pprint
import sys

import matplotlib.pyplot as plt
from spdm.numlib import np
from spdm.numlib import constants as constants


if __name__ == "__main__":

    from spdm.data.Document import Document
    from spdm.data.Collection import Collection
    from spdm.data.File import File
    from spdm.util.logger import logger

    doc = Collection("EAST+mdsplus:///home/salmon/public_data/~t/?tree_name=efit_east")\
        .open(shot=55555, time_slice=20, mode="r")
    # for coil in entry.pf_active.coil:
    #     logger.debug(coil.current.__value__())
    # entry = Document({"path": ["/home/salmon/workspace/fytok/devices/EAST/imas/3/dynamic/config.xml",
    #                            "/home/salmon/workspace/fytok/devices/EAST/imas/3/static/config.xml"
    #                            ],
    #                   "schema": "file/XML"}).entry
    # doc = File(path=["/home/salmon/workspace/fytok/devices/EAST/imas/3/dynamic/config.xml",
    #                  "/home/salmon/workspace/fytok/devices/EAST/imas/3/static/config.xml"],
    #            file_format=".xml")
    for coil in doc.entry.pf_active.coil:
        logger.debug(coil.current.data.__value__())
