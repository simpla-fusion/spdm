import numpy as np
from ..util.logger import logger


def float_unique(d: np.ndarray, x_min=-np.inf, x_max=np.inf) -> np.ndarray:
    if not isinstance(d, np.ndarray):
        d = np.asarray(d)
    d = np.sort(d)
    tag = np.append(True, np.diff(d)) > np.finfo(float).eps*10
    rtag = np.logical_and(d >= x_min, d <= x_max)
    rtag = np.logical_or(rtag, np.isclose(d, x_min, rtol=1e-8))
    rtag = np.logical_or(rtag, np.isclose(d, x_max, rtol=1e-8))
    tag = np.logical_and(tag, rtag)
    return d[tag]
