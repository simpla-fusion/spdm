import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import scipy.stats


if __name__ == "__main__":

    sys.path.append("/home/salmon/workspace/SpDev/SpDB")

    from spdm.data.File import File
    from spdm.util.logger import logger

    fp = File("/home/salmon/workspace/SpDev/SpDB/examples/data/FuYun/modules/physics/genray/template/g063982.04800",
              metadata={"$schema": "file/geqdsk"})

    logger.debug(fp)
