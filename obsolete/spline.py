from __future__ import division, print_function, absolute_import
from ..util.logger import logger
from scipy import interpolate
import numpy as np
from scipy.interpolate import PPoly, CubicSpline


def create_spline(x, y, **kwargs) -> PPoly:
    bc_type = "periodic" if np.all(y[0] == y[-1]) else "not-a-knot"
    try:
        res = CubicSpline(x, y, bc_type=bc_type)
    except ValueError as error:
        logger.error((x, y))
        raise error
    return res


def create_spline_for_bvp(y, yp, x, dx):
    """Create a cubic spline given values and derivatives.

    Formulas for the coefficients are taken from interpolate.CubicSpline.

    Returns
    -------
    sol : PPoly
        Constructed spline as a PPoly instance.
    """
    from scipy.interpolate import PPoly

    n, m = y.shape
    c = np.empty((4, n, m - 1), dtype=y.dtype)
    slope = (y[:, 1:] - y[:, :-1]) / dx
    t = (yp[:, :-1] + yp[:, 1:] - 2 * slope) / dx
    c[0] = t / dx
    c[1] = (slope - yp[:, :-1]) / dx - t
    c[2] = yp[:, :-1]
    c[3] = y[:, :-1]
    c = np.rollaxis(c, 1)

    return PPoly(c, x, extrapolate=True, axis=1)
