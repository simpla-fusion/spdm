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
                    handlers=[logging.FileHandler(SP_DEFAULT_OUTPUT_DIR/f"sp_{os.getpid()}.log"),
                              logging.StreamHandler(stream=sys.stdout)
                              ]
                    )
logger = logging.getLogger(__package__[:__package__.find('.')])
logger.setLevel(logging.DEBUG)
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
