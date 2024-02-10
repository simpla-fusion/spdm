import sys
import numpy as np
import timeit

sys.path.append("/home/salmon/workspace/SpDev/SpDB")

if __name__ == "__main__":
    from ..util.logger import logger
    from spdm.core.Profile import Profile, Profiles

    npoints = 100

    x_axis = np.linspace(0, 1.0, npoints)

    x, y, z = np.random.rand(3, npoints)

    X = x.view(Profile)

    X._axis = x_axis
    Y = y.view(Profile)
    Y._axis = x_axis
    Z = z.view(Profile)
    Z._axis = x_axis

    r = x + (y**2 + (z*x + 1)*3)
    R = X + (Y**2 + (Z*X + 1)*3)

    logger.debug(R.description)

    logger.debug(all(r == R))

    time0 = timeit.timeit(lambda: x + (y**2 + (z*x + 1)*3), number=100)
    time1 = timeit.timeit(lambda: R.evaluate(), number=100)

    logger.debug((time0, time1))
    
    logger.debug(R.derivative)