from __future__ import division, print_function, absolute_import
from ..util.logger import logger
from scipy import interpolate
from spdm.numlib import np
from scipy.interpolate import PPoly, CubicSpline


def create_spline(x, y, **kwargs) -> PPoly:
    bc_type = "periodic" if np.all(y[0] == y[-1]) else "not-a-knot"
    return CubicSpline(x, y, bc_type=bc_type)


def create_spline_for_bvp(y, yp, x, dx, discontinuity=[]):
    """Create a cubic spline given values and derivatives.

    Formulas for the coefficients are taken from interpolate.CubicSpline.

    discontinuity: list *experimental*
            NOTE: add by salmon 
            List of discontinuity points
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
    # ###########################
    # # add by salmon
    # if discontinuity is None:
    #     discontinuity = []
    # for ix in discontinuity:
    #     idx = np.argmax(x >= ix)
    #     if idx == 0:
    #         rms_res[0] = rms_res[1]
    #     elif idx < len(rms_res)-1:
    #         rms_res[idx-2] = rms_res[idx-3]
    #         rms_res[idx-1] = rms_res[idx-2]  # (rms_res[idx-2] + rms_res[idx+1])*0.5
    #         rms_res[idx] = rms_res[idx+1]

    # ###########################
    return PPoly(c, x, extrapolate=True, axis=1)
