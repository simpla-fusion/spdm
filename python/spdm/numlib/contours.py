from .common import np
from ..util.logger import logger
import typing
import matplotlib.pyplot as plt

# d: np.ndarray, x: typing.Optional[np.ndarray] = None, y: typing.Optional[np.ndarray] = None


def find_countours(*args,  ** kwargs) -> typing.List[typing.List[np.ndarray]]:
    """
        args:X: np.ndarray, Y: np.ndarray, Z: np.ndarray
        TODO: need improvement
    """
    fig = plt.figure()
    contour_set = fig.gca().contour(*args, ** kwargs)
    return [(contour_set.levels[idx], col.get_segments()) for idx, col in enumerate(contour_set.collections)]
