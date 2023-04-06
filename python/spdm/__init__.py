__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from .version import __version__

# from .data.open_entry import open_entry, open_db

# import numpy as np
# import scipy
# from .common.logger import logger
# logger.info(f"Using SciPy \t: {scipy.__version__}")
# logger.info(f"Using NumPy \t: {np.version.full_version}")
# logger.info(f"Using SpDB  \t: {__version__}")
# ENABLE_JAX = os.environ.get("SP_JAX", False)
# if ENABLE_JAX:
#     import jax
#     import jax.numpy as np
#     from jax.scipy.optimize import minimize
#     logger.info(f"Using JAX \t: {jax.__version__}")
# else:
#     import scipy.interpolate as interpolate
#     from scipy.optimize import fsolve, minimize, root_scalar
