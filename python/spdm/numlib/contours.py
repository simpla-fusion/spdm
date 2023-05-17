import typing

import numpy as np
import scipy.interpolate as interpolate
from skimage import measure

from ..data.Field import Field
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


def find_countours_skimage(z: np.ndarray, x: np.ndarray = None, y: np.ndarray = None,  levels=128) -> typing.Generator[typing.Tuple[float, np.ndarray], None, None]:
    if z.shape == x.shape and z.shape == y.shape:
        pass
    else:
        raise ValueError(f"Array shape does not match! x:{x.shape} , y:{y.shape}, z:{z.shape} ")
    shape = z.shape
    dim0 = np.linspace(0, shape[0]-1, shape[0])
    dim1 = np.linspace(0, shape[1]-1, shape[1])
    x_inter = interpolate.RectBivariateSpline(dim0, dim1, x)
    y_inter = interpolate.RectBivariateSpline(dim0, dim1, y)

    if isinstance(levels, (int, np.integer)):
        levels = range(levels)
    elif (isinstance(levels, (np.ndarray)) and levels.ndim == 0) or isinstance(levels, (float, np.floating)):
        levels = [float(levels)]
  
    for val in levels:
        for c in measure.find_contours(z, val):
            # data = [[x_inter(p[0], p[1], grid=False), y_inter(p[0], p[1], grid=False)] for p in c]
            x = np.asarray(x_inter(c[:, 0], c[:, 1], grid=False), dtype=float)
            y = np.asarray(y_inter(c[:, 0], c[:, 1], grid=False), dtype=float)
            data = np.stack([x, y], axis=-1)
            yield val,  data


def find_countours(*args, levels=128) -> typing.Generator[typing.Tuple[float, np.ndarray], None, None]:
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

    yield from find_countours_skimage(z, x, y, levels=levels)
