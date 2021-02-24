
import collections
import copy
import inspect
import numbers
import types
from functools import cached_property, lru_cache

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.constants
import scipy.integrate
import scipy.misc
from scipy.interpolate import RectBivariateSpline, UnivariateSpline, interp1d
from spdm.util.AttributeTree import AttributeTree
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from .DataObject import DataObject


class PhysicalQuantity(np.nparray, DataObject):
    """
    """
    pass
