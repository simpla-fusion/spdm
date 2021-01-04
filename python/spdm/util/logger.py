import logging
import logging.handlers
import inspect
import pathlib
import pprint
import os
import sys

SP_DEFAULT_OUTPUT_DIR = pathlib.Path("~/spdm_log").expanduser()
SP_DEFAULT_OUTPUT_DIR.mkdir(mode=0o755, exist_ok=True)

std_handler = logging.StreamHandler(stream=sys.stdout)


logging.basicConfig(level=logging.ERROR,
                    format=(
                        '%(asctime)s %(levelname)s [%(name)s] '
                        '%(pathname)s:%(lineno)d:%(funcName)s: '
                        '%(message)s'),
                    handlers=[logging.FileHandler(SP_DEFAULT_OUTPUT_DIR/f"sp_{os.getpid()}.log") ]
                    )
logger = logging.getLogger(__package__[:__package__.find('.')])
logger.setLevel(logging.DEBUG)


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[0;37m"
    yellow = "\x1b[1;33m"
    green = "\x1b[0;32m"
    red = "\x1b[0;31m"
    bold_red = "\x1b[1;31m"
    reset = "\x1b[0m"
    format_normal =  '%(asctime)s %(levelname)s [%(name)s] %(pathname)s:%(lineno)d:%(funcName)s: %(message)s'

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

# create console handler and set level to debug
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
# add formatter to ch
ch.setFormatter(CustomFormatter())
# add ch to logger
logger.addHandler(ch)

# logging.getLogger('matplotlib').setLevel(logging.ERROR)


# TODO (salmon 20190922): support more log handlers,


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
