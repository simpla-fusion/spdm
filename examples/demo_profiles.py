
import sys

import numpy as np

sys.path.append("/home/salmon/workspace/freegs/")
sys.path.append("/home/salmon/workspace/fytok/python")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")

if __name__ == "__main__":
    from spdm.util.logger import logger
    from spdm.util.Profiles import Profile, Profiles

    x_axis = np.linspace(0, 1.0, 128)
    a = Profile(np.sqrt, x_axis, description={"name": "a"})
    logger.debug("#########################################")
    # b = Profile(np.sqrt, x_axis)

    c = a[1:5]
    # logger.debug(c.description)
    # logger.debug(type(c))
    logger.debug(c._x_axis is x_axis)
    logger.debug(c._x_axis == x_axis[1:5])

    # d = c[1:5]
    # logger.debug(type(d))
    # logger.debug(d._x_axis is x_axis)
