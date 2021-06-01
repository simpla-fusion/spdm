__path__ = __import__('pkgutil').extend_path(__path__, __name__)
__version__= '0.0.0'


from . import data, flow  # ,module, numerical, util
from .util.logger import logger

logger.info(f"Using SpDB \t: {__version__}")
