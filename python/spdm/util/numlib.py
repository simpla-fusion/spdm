import numpy as np
import scipy
import scipy.interpolate as interpolate
import scipy.constants as constants
import scipy.optimize as optimize
from .logger import logger
_array_cls = np.ndarray

logger.debug(f"Using SciPy: {scipy.__version__}")
logger.debug(f"Using NumPy: {np.version.full_version}")
