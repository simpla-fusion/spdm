__path__ = __import__('pkgutil').extend_path(__path__, __name__)

# try:
#     from .__version__ import __version__
# except:
#     # try:
#     #     import subprocess
#     #     __version__ = subprocess.check_output(['git', 'describe', '--always', '--dirty']).strip().decode('utf-8')
#     # except:
__version__ = "0.0.0"
ENABLE_JAX = True  # os.environ.get("SP_ENABLE_JAX", False)

# if ENABLE_JAX:
#     from jax import numpy
#     from jax import scipy
#     # from jax.scipy.optimize import minimize
#     # logger.info(f"Using JAX \t: {jax.__version__}")
# else:
import numpy
import scipy
    # import scipy.constants as constants
    # import scipy.interpolate as interpolate
    # from scipy.optimize import fsolve, minimize, root_scalar

#     logger.info(f"Using SciPy \t: {scipy.__version__}")
#     logger.info(f"Using NumPy \t: {np.version.full_version}")
