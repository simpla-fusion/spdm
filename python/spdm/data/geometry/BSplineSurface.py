from functools import cached_property
from operator import is_

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import make_interp_spline

from ...util.logger import logger
from .Surface import Surface
from ..Function import Function


class BSplineSurface(Surface):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        raise NotImplementedError()
