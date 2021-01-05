import logging
import logging.handlers
import inspect
import pathlib
import pprint
import os
import sys
from datetime import datetime


logger = logging.getLogger(__package__[:__package__.find('.')])

logger.setLevel(logging.DEBUG)

default_formater = logging.Formatter('%(asctime)s %(levelname)s [%(name)s] '
                                     '%(pathname)s:%(lineno)d:%(funcName)s: '
                                     '%(message)s')


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[0;37m"
    yellow = "\x1b[1;33m"
    green = "\x1b[0;32m"
    red = "\x1b[0;31m"
    bold_red = "\x1b[1;31m"
    reset = "\x1b[0m"
    format_normal = '%(asctime)s %(levelname)s [%(name)s] %(pathname)s:%(lineno)d:%(funcName)s: %(message)s'

    FORMATS = {
        logging.DEBUG: grey + format_normal + reset,
        logging.INFO:  green + '%(asctime)s %(levelname)s [%(name)s] : %(message)s' + reset,
        logging.WARNING: yellow + format_normal + reset,
        logging.ERROR: red + format_normal + reset,
        logging.CRITICAL: bold_red + format_normal + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def sp_enable_logging(handler=None, prefix=None, formater=None):

    if isinstance(handler, str) and handler == "STDOUT":
        handler = logging.StreamHandler(stream=sys.stdout)
        formater = formater or CustomFormatter()
    elif handler is None:
        path = pathlib.Path(
            f"{prefix or os.environ.get('SP_LOG_PREFIX', '/tmp/sp_log/sp_')}{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.log")
        path = path.expanduser().resolve()
        if not path.parent.exists():
            path.parent.mkdir(mode=0o0755, exist_ok=True)
        handler = logging.FileHandler(path)

    if issubclass(type(handler), logging.Handler):
        handler.setFormatter(formater or default_formater)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    else:
        raise NotImplementedError()


if not os.environ.get("SP_NO_DEBUG", None):
    sp_enable_logging("STDOUT")
    # add_logging_handler()


def deprecated(func):

    def _wrap(func):
        def wrapped(*args, __fun__=func, ** kwargs):

            if inspect.isfunction(func):
                logger.warning(
                    f"Deprecated function '{__fun__.__qualname__}' !")
                raise DeprecationWarning(__fun__.__qualname__)
            else:
                logger.warning(f"Deprecated object {__fun__}")
            return __fun__(*args, **kwargs)
        return wrapped

    if func is None:
        return lambda o: _wrap(func)
    else:
        return _wrap(func)
