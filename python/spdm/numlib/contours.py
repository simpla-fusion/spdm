import typing

import numpy as np
import collections.abc
import scipy.interpolate
from skimage import measure

from ..data.Field import Field
from ..geometry.Curve import Curve
from ..geometry.GeoObject import GeoObject
from ..geometry.Point import Point
from ..utils.logger import deprecated, logger

# import matplotlib.pyplot as plt
# @deprecated
# def find_countours_matplotlib(z: np.ndarray, x: np.ndarray = None, y: np.ndarray = None, /, *args, levels=None, ** kwargs) -> typing.List[typing.List[np.ndarray]]:
#     """
#         args:X: np.ndarray, Y: np.ndarray, Z: np.ndarray
#         TODO: need improvement
#     """
#     fig = plt.figure()
#     contour_set = fig.gca().contour(x, y, z, *args, levels=levels, ** kwargs)
#     return [(contour_set.levels[idx], col.get_segments()) for idx, col in enumerate(contour_set.collections)]


def find_countours_skimage_(val: float, z: np.ndarray, x_inter, y_inter) -> typing.Generator[GeoObject | None, None, None]:
  
    for c in measure.find_contours(z, val):
        # data = [[x_inter(p[0], p[1], grid=False), y_inter(p[0], p[1], grid=False)] for p in c]
        x = np.asarray(x_inter(c[:, 0], c[:, 1], grid=False), dtype=float)
        y = np.asarray(y_inter(c[:, 0], c[:, 1], grid=False), dtype=float)
        data = np.stack([x, y], axis=-1)
        
        if len(data) == 0:
            yield None
        elif data.shape[0] == 1:
            yield Point(*data[0])
        else:
            yield Curve(data)


def find_countours_skimage(vals: list, z: np.ndarray, x: np.ndarray, y: np.ndarray):
    if z.shape == x.shape and z.shape == y.shape:
        pass
    else:
        raise ValueError(f"Array shape does not match! x:{x.shape} , y:{y.shape}, z:{z.shape} ")
    shape = z.shape
    dim0 = np.linspace(0, shape[0]-1, shape[0])
    dim1 = np.linspace(0, shape[1]-1, shape[1])
    x_inter = scipy.interpolate.RectBivariateSpline(dim0, dim1, x)
    y_inter = scipy.interpolate.RectBivariateSpline(dim0, dim1, y)

    if not isinstance(vals, (collections.abc.Sequence, np.ndarray)):
        vals = [vals]

    for val in vals:
        yield val, find_countours_skimage_(val, z, x_inter, y_inter)

        # count = 0
        # for c in measure.find_contours(z, val):
        #     count += 1
        #     # data = [[x_inter(p[0], p[1], grid=False), y_inter(p[0], p[1], grid=False)] for p in c]
        #     x = np.asarray(x_inter(c[:, 0], c[:, 1], grid=False), dtype=float)
        #     y = np.asarray(y_inter(c[:, 0], c[:, 1], grid=False), dtype=float)
        #     data = np.stack([x, y], axis=-1)
        #     if len(data) == 0:
        #         yield val, None
        #     elif data.shape[0] == 1:
        #         yield val, Point(*data[0])
        #     else:
        #         yield val, Curve(data)
        # if count == 0:
        #     yield val, None


def find_countours(*args, values, ** kwargs) -> typing.Generator[typing.Tuple[float,  typing.Generator[GeoObject | None, None, None]], None, None]:
    if len(args) == 3:
        z, x, y = args
    elif len(args) == 1:
        if not isinstance(args[0], Field):
            raise TypeError(f"Wrong type of argument! should be Field, got {type(args[0])}")
        f = args[0]
        xy = f.mesh.points
        x = xy[0]
        y = xy[1]
        z = np.asarray(f)
    else:
        raise ValueError(f"Wrong number of arguments! should be 1 or 3, got {len(args)}")

    yield from find_countours_skimage(values, z, x, y, **kwargs)
