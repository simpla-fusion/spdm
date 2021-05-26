from functools import cached_property, lru_cache

from spdm.util.numlib import np
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure
from scipy.optimize import fsolve, root_scalar

from ..util.logger import logger
from .Mesh import Mesh
import pprint


def find_peaks_2d(Z):
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
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # local_extremum = binary_erosion(local_extremum, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_extremum mask (xor operation)
    detected_peaks = local_extremum ^ eroded_background
    idxs = np.asarray(np.where(detected_peaks)).T

    for ix, iy in idxs:
        yield ix, iy


class StructuredMesh(Mesh):
    def __init__(self, *args,  **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def axis(self, idx, axis=0):
        return NotImplemented

    def remesh(self, *arg, **kwargs):
        return NotImplemented

    def interpolator(self, Z):
        return NotImplemented


    def find_peak(self, Z):

        X, Y = self.points

        func = self.interpolator(Z)

        fxy2 = (func(X.ravel(), Y.ravel(), dx=1, grid=False)**2 +
                func(X.ravel(), Y.ravel(), dy=1, grid=False)**2).reshape(Z.shape)

        for ix, iy in find_peaks_2d(fxy2):
            if ix == 0 or iy == 0 or ix == Z.shape[0]-1 or iy == Z.shape[1]-1:
                continue

            xmin = X[ix-1, iy]
            xmax = X[ix+1, iy]
            ymin = Y[ix, iy-1]
            ymax = Y[ix, iy+1]

            def f(r):
                if r[0] < xmin or r[0] > xmax or r[1] < ymin or r[1] > ymax:
                    raise LookupError(r)
                fx = func(r[0], r[1], dx=1, grid=False)
                fy = func(r[0], r[1], dy=1, grid=False)
                return fx, fy

            def fprime(r):
                fxx = func(r[0], r[1], dx=2, grid=False)
                fyy = func(r[0], r[1], dy=2, grid=False)
                fxy = func(r[0], r[1], dy=1, dx=1, grid=False)

                return [[fxx, fxy], [fxy, fyy]] 

            x = X[ix, iy]
            y = Y[ix, iy]
            try:
                x1, y1 = fsolve(f, [x, y],   fprime=fprime)
            except LookupError as error:
                # TODO: need handle exception
                # logger.debug(error)
                continue
            else:
                x = x1
                y = y1

            D = func(x, y, dx=2, grid=False) * func(x, y, dy=2, grid=False) - (func(x, y,  dx=1, dy=1, grid=False))**2

            yield x, y, func(x, y, grid=False), D
