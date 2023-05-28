import collections.abc
import os
import pprint
import typing
import numpy as np
from scipy import optimize
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure
from scipy.optimize import fsolve, minimize, root_scalar
from ..utils.logger import logger
from ..data.Field import Field
from ..utils.typing import NumericType, ScalarType, ArrayType
SP_EXPERIMENTAL = os.environ.get("SP_EXPERIMENTAL", False)

# logger.info(f"SP_EXPERIMENTAL \t: {SP_EXPERIMENTAL}")

EPSILON = 1.0e-2


def _minimize_filter_2d_image(Z) -> typing.Generator[typing.Tuple[int, int], None, None]:
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    # local_extremum = (maximum_filter(Z, footprint=neighborhood) == Z)\
    local_extremum = (minimum_filter(Z, footprint=neighborhood) == Z)

    # local_extremum is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (Z == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_extremum, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1)

    # local_extremum = binary_erosion(local_extremum, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_extremum mask (xor operation)
    detected_peaks = local_extremum ^ eroded_background
    idxs = np.asarray(np.where(detected_peaks)).T

    for ix, iy in idxs:
        yield ix, iy


def minimize_filter(func: typing.Callable[..., ScalarType], xmin: float, ymin: float, xmax: float, ymax: float, tolerance=EPSILON):

    if isinstance(tolerance, float):
        dx = tolerance
        dy = tolerance
    elif isinstance(tolerance, (collections.abc.Sequence, np.ndarray)) and len(tolerance) == 2:
        dx, dy = tolerance
    else:
        raise TypeError(f"Illegal type {type(dx)}")

    nx = int((xmax-xmin)/dx)+1
    ny = int((ymax-ymin)/dy)+1

    X, Y = np.meshgrid(np.linspace(xmin, xmax, nx),
                       np.linspace(ymin, ymax, ny), indexing='ij')
    
    data = func(X, Y)

    for ix, iy in _minimize_filter_2d_image(data):

        if ix == 0 or iy == 0 or ix == nx-1 or iy == ny-1:
            continue

        xmin = X[ix-1, iy]
        xmax = X[ix+1, iy]
        ymin = Y[ix, iy-1]
        ymax = Y[ix, iy+1]

        xsol = X[ix, iy]
        ysol = Y[ix, iy]

        if True:
            sol = minimize(lambda x: func(x[0], x[1]), np.asarray([xsol, ysol]),   bounds=[(xmin, xmax), (ymin, ymax)])
            xsol, ysol = sol.x
            if sol.success and abs(sol.fun) < EPSILON and (xsol >= xmin or xsol <= xmax or ysol >= ymin or ysol <= ymax):
                yield xsol, ysol
        else:  # obsolete
            def f(r):
                if r[0] < xmin or r[0] > xmax or r[1] < ymin or r[1] > ymax:
                    raise LookupError(r)
                fx = func(r[0], r[1], dx=1, mesh=False)
                fy = func(r[0], r[1], dy=1, mesh=False)
                return fx, fy

            def fprime(r):
                fxx = func(r[0], r[1], dx=2, mesh=False)
                fyy = func(r[0], r[1], dy=2, mesh=False)
                fxy = func(r[0], r[1], dy=1, dx=1, mesh=False)

                return [[fxx, fxy], [fxy, fyy]]

            try:
                x1, y1 = fsolve(f, [xsol, ysol],   fprime=fprime)
            except LookupError as error:
                # TODO: need handle exception
                # logger.debug(error)
                continue
            else:
                xsol = x1
                ysol = y1


def minimize_filter_2d_experimental(func: typing.Callable[..., ScalarType], x0, y0, x1, y1, p0=None, tolerance=[0.1, 0.1]):
    if abs((x0-x1)) < tolerance[0] or abs(y0-y1) < tolerance[1]:
        yield from []
        return

    xmin = min(x0, x1)
    xmax = max(x0, x1)
    ymin = min(y0, y1)
    ymax = max(y0, y1)
    xc = 0.5*(x0 + x1)
    yc = 0.5*(y0 + y1)

    if p0 is None or not (xmin < p0[0] and xmax > p0[0] and ymin < p0[1] and ymax > p0[1]):
        # ,   bounds=[(xmin, xmax), (ymin, ymax)]
        sol = minimize(func, np.asarray([xc, yc]), method="BFGS")
        p0 = sol.x
        if sol.success and (xmin < p0[0] and xmax > p0[0] and ymin < p0[1] and ymax > p0[1]):
            yield p0[0], p0[1]
            logger.debug((p0, (xmin, ymin), (xmax, ymax)))
        else:
            p0 = None

    yield from minimize_filter_2d_experimental(func, xc, yc, xmin, ymax, p0=p0, tolerance=tolerance)
    yield from minimize_filter_2d_experimental(func, xc, yc, xmin, ymin, p0=p0, tolerance=tolerance)
    yield from minimize_filter_2d_experimental(func, xc, yc, xmax, ymin, p0=p0, tolerance=tolerance)
    yield from minimize_filter_2d_experimental(func, xc, yc, xmax, ymax, p0=p0, tolerance=tolerance)


def find_critical_points_2d_experimental(func: Field, xmin, ymin, xmax, ymax, tolerance=EPSILON):

    # def grad_func(p): return func(*p, dx=1, mesh=False)**2 + func(*p, dy=1, mesh=False)**2

    def grad_func(*x): return func(*x, mesh=False)

    for x, y in minimize_filter_2d_experimental(grad_func, xmin, ymin, xmax, ymax, tolerance=tolerance):
        D = func.pd(2, 0)(x, y,  mesh=False) * func.pd(0, 2)(x, y, mesh=False) - \
            (func(x, y,  dx=1, dy=1, mesh=False))**2
        yield x, y, func(x, y, mesh=False), D


def find_critical_points(f: Field, tolerance=EPSILON):
    assert (isinstance(f, Field))

    xmin, ymin = f.mesh.geometry.bbox[0]
    xmax, ymax = f.mesh.geometry.bbox[1]

    if tolerance is None:
        tolerance = f.dx

    grad_fun = f.pd(1, 0) ** 2 + f.pd(0, 1)**2

    for xsol, ysol in minimize_filter(grad_fun, xmin, ymin, xmax, ymax, tolerance=tolerance):
        D = f.pd(2, 0)(xsol, ysol) * f.pd(0, 2)(xsol, ysol) - \
            (f.pd(1, 1)(xsol, ysol))**2
        v = f(xsol, ysol)
        if isinstance(v, np.ndarray) and v.ndim == 0:
            v = float(v)
        yield xsol, ysol, v, D
