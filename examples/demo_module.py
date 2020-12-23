import functools
import collections
import matplotlib.pyplot as pltimport
import numpy as np
import sys
import pprint
sys.path.append("/home/salmon/workspace/SpDev/SpDB")


if __name__ == "__main__":

    from spdm.util.logger import logger
    from spdm.util.ModuleManager import ModuleManager

    module = ModuleManager()

    module.load_configure("/home/salmon/workspace/SpDev/SpDB/examples/data/configure.yaml")

    module.factory.add_alias("/actors/", "PyObject:///spdm/util/*#{fragment}")


    genray = module.create("/modules/physics/genray")

    output0 = genray(num_of_steps=1)

    # output2 = cql3d(num_of_steps=1, equilibrium=output0.eq)
    logger.debug(output0)

    logger.debug(pprint.pformat([p for p in module.glob()]))
