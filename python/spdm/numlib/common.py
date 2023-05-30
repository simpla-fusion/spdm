import os

# from spdm.utils.logger import logger

ENABLE_JAX = False  # os.environ.get("SP_ENABLE_JAX", False)

if ENABLE_JAX:
    from jax import numpy
    from jax import scipy
    # from jax.scipy.optimize import minimize
    # logger.info(f"Using JAX \t: {jax.__version__}")
else:
    import numpy
    import scipy
    # import scipy.constants as constants
    # import scipy.interpolate as interpolate
    # from scipy.optimize import fsolve, minimize, root_scalar

#     logger.info(f"Using SciPy \t: {scipy.__version__}")
#     logger.info(f"Using NumPy \t: {np.version.full_version}")
