from .logger import logger
import os
import scipy

import scipy.constants as constants

ENABLE_JAX = os.environ.get("SP_JAX", False)

if ENABLE_JAX:
    import jax
    import jax.numpy as np
    from jax.scipy.optimize import minimize

    logger.info(f"Using JAX \t: {jax.__version__}")
else:
    import numpy as np
    from scipy.optimize import minimize, root_scalar
    import scipy.interpolate as interpolate

    logger.info(f"Using SciPy \t: {scipy.__version__}")
    logger.info(f"Using NumPy \t: {np.version.full_version}")

_array_cls = np.ndarray
