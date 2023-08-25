
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from spdm.utils.plot_profiles import plot_profiles
from spdm.data import Function
from ..util.logger import logger

if __name__ == '__main__':

    x = np.linspace(0, constants.pi*2.0, 64, endpoint=True)
    y = np.sin(x)
    m = np.zeros([3, 64])
    # f0 = Function(x, y, is_periodic=True)
    # f1 = Function(np.linspace(0, 1.0, 128), y, is_periodic=True)
    f1 = Function(x, [(lambda t:t < 1), (lambda t:t >= 1)], [lambda t:t**2, lambda t:0.4*np.sin(t)])
    # f2 = (f1*2+x)*m
    # logger.debug(f2.shape)
    x2 = np.linspace(0, constants.pi*2.0, 128, endpoint=True)
    f3 = Function(x, np.sin)

    d = (f3*2+f1*x)(x2)

    plot_profiles([
        # [(f0, r"$f0(x)$"),
        #  (f1, r"$f1(x)=f0(2\pi x)$")],
        # [(np.cos(x), r"$cos(x)$"),
        #  (f0.derivative, r"$d f0(x)$"),
        #  (f1.derivative/(constants.pi*2.0), r"$d f1(x)/2\pi$")],
        # [(-np.cos(x)+np.cos(0), r"-$cos(x)+cos(0)$"),
        #  (f0.antiderivative, r"$\int f0$"),
        #  (constants.pi*2.0*f1.antiderivative, r"2\pi $\int f1$")],
        (f1,  "f1"),
        # (f2,  "f2"),
        (f3,  "f3")
    ],
        x_axis=(x2, "x")
    ).savefig("/home/salmon/workspace/output/test_function.svg")
