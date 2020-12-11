
import sys

import numpy as np

sys.path.append("/home/salmon/workspace/freegs/")
sys.path.append("/home/salmon/workspace/fytok/python")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")

if __name__ == "__main__":
    from spdm.util.logger import logger
    from spdm.util.Profiles import Profile, Profiles

    # axis = np.linspace(0, 1.0, 128)
    # a = Profile(np.sqrt, axis, description={"name": "a"})
    # logger.debug("#########################################")
    # # b = Profile(np.sqrt, axis)

    # c = a[1:5]
    # # logger.debug(c.description)
    # # logger.debug(type(c))
    # logger.debug(c._axis is axis)
    # logger.debug(c._axis == axis[1:5])
    npoints = 11
    x0 = np.linspace(0, 1.0, npoints)
    x1 = np.linspace(0, 1.0, npoints)

    rho_b = 0.95

    def D(r): return np.piecewise(r, [r < rho_b, r >= rho_b], [lambda x: (0.5 + (x**3)), 0.1])

    a = Profile(D, axis=x0)
    b = Profile(np.linspace(0, 5, npoints), axis=x1)
    b += a
    logger.debug(b)
