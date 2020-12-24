import functools
import collections
import matplotlib.pyplot as pltimport
import numpy as np
import sys
import os
import pprint
sys.path.append("/home/salmon/workspace/SpDev/SpDB")


if __name__ == "__main__":

    from spdm.util.logger import logger
    from spdm.util.ModuleRepository import ModuleRepository

    os.environ["FUYUN_CONFIGURE_PATH"] = "/home/salmon/workspace/SpDev/SpDB/examples/data/FuYun/configure.yaml"

    module = ModuleRepository(repo_name='FuYun', repo_tag='FY')

    module.factory.add_alias("/actors/", "PyObject:///spdm/actors/*#{fragment}")

    logger.debug(pprint.pformat(module.factory.alias._mmap))
    logger.debug(pprint.pformat(module.resolver.alias._mmap))

    genray = module.create("/modules/physics/genray", version="10.8", tag_suffix="-foss-2019", workingdir="./")

    output0 = genray(num_of_steps=1)

    # output2 = cql3d(num_of_steps=1, equilibrium=output0.eq)
    logger.debug((genray._description))

    logger.debug(pprint.pformat([p for p in module.glob()]))
