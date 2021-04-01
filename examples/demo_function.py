
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
from spdm.util.plot_profiles import plot_profiles
from spdm.data.Function import Function
from spdm.util.logger import logger

if __name__ == '__main__':

    x = np.linspace(0, scipy.constants.pi*2.0, 128, endpoint=True)
    y = np.sin(x)

    f0 = Function(x, y, is_periodic=True)
    f1 = Function(np.linspace(0, 1.0, 128), y, is_periodic=True)
    f2 = f1*2
    logger.debug(type(f2))
    plot_profiles([
        [(f0, r"$f0(x)$"),
         (f1, r"$f1(x)=f0(2\pi x)$")],
        [(np.cos(x), r"$cos(x)$"),
         (f0.derivative, r"$d f0(x)$"),
         (f1.derivative/(scipy.constants.pi*2.0), r"$d f1(x)/2\pi$")],
        [(-np.cos(x)+np.cos(0), r"-$cos(x)+cos(0)$"),
         (f0.antiderivative, r"$\int f0$"),
         (scipy.constants.pi*2.0*f1.antiderivative, r"2\pi $\int f1$")]

    ],
        # x_axis=(x, "x")
    ).savefig("/home/salmon/workspace/output/test_function.svg")
