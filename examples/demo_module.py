import functools
import collections
import matplotlib.pyplot as pltimport
import numpy as np
import sys
import pprint
sys.path.append("/home/salmon/workspace/SpDev/SpDB")


if __name__ == "__main__":

    from spdm.util.logger import logger
    from spdm.util.ModuleRepository import ModuleRepository

    module = ModuleRepository()

    module.load_configure("/home/salmon/workspace/SpDev/SpDB/examples/data/configure.yaml")

    module.factory.add_alias("/actors/", "PyObject:///spdm/actors/*#{fragment}")

    genray = module.create("/modules/physics/genray", version="10.8", tag_suffix="foss-2019", workingdir="./")

    output0 = genray(num_of_steps=1)

    # output2 = cql3d(num_of_steps=1, equilibrium=output0.eq)
    logger.debug(type(genray))

    logger.debug(pprint.pformat([p for p in module.glob()]))
