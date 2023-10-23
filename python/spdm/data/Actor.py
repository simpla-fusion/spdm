import tempfile
import shutil
import pathlib
import os
import contextlib
from ..view import View as sp_view
from ..utils.logger import logger
from ..utils.plugin import Pluggable
from ..utils.envs import SP_MPI, SP_DEBUG
from .sp_property import SpTree
import getpass


class Actor(SpTree, Pluggable):
    mpi_enabled = False
    _plugin_prefix = __package__
    _plugin_registry = {}

    def __init__(self, *args, **kwargs):
        if self.__class__ is Actor or "_plugin_prefix" in vars(self.__class__):
            self.__class__.__dispatch_init__(None, self, *args, **kwargs)
            return
        super().__init__(*args, **kwargs)

    @property
    def tag(self) -> str: return f"{self._plugin_prefix}{self.__class__.__name__.lower()}"

    @property
    def MPI(self): return SP_MPI

    def _repr_svg_(self) -> str:
        try:
            res = sp_view.display(self, output="svg")
        except Exception as error:
            logger.error(error)
            res = None
        return res

    def __geometry__(self,  *args,  **kwargs):
        return {}

    @contextlib.contextmanager
    def working_dir(self):
        temp_dir = None
        if SP_DEBUG:
            _working_dir = pathlib.Path(f"{self.output_dir}/{self.tag}")
            _working_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix=self.tag)
            _working_dir = pathlib.Path(temp_dir.name)

        pwd = os.getcwd()

        os.chdir(_working_dir)

        logger.info(f"Enter directory {_working_dir}")

        error = None

        try:
            yield _working_dir
        except Exception as e:
            error = e

        if (error is not None and temp_dir is not None):
            shutil.copytree(temp_dir.name, f"{self.output_dir}/{self.tag}", dirs_exist_ok=True)
        elif temp_dir is not None:
            temp_dir.cleanup()

        os.chdir(pwd)
        logger.info(f"Enter directory {pwd}")

        if error is not None:
            raise RuntimeError(
                f"Failed to execute actor {self.tag}! see log in {self.output_dir}/{self.tag}") from error

    @property
    def output_dir(self) -> str: return self.get("output_dir", None) or os.getenv("SP_OUTPUT_DIR", None) or os.getcwd()
