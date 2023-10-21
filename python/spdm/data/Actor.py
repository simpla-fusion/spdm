import tempfile
import shutil
import pathlib
import os
from ..view import View as sp_view
from ..utils.logger import logger
from ..utils.plugin import Pluggable
from ..utils.envs import SP_MPI, SP_DEBUG
from .sp_property import SpTree
import getpass


class Actor(SpTree, Pluggable):
    mpi_enabled = False

    _plugin_registry = {}

    def __init__(self, *args, **kwargs):
        if self.__class__ is Actor or "_plugin_prefix" in vars(self.__class__):
            self.__class__.__dispatch_init__(None, self, *args, **kwargs)
            return
        super().__init__(*args, **kwargs)
        self._working_dir = None
        self._log_dir = kwargs.get("log_dir", None) or f"{os.getcwd()}/{self.tag}/"

    @property
    def tag(self) -> str:
        return f"{os.getpid()}_{self.__class__.__name__.lower()}_{getpass.getuser()}_{os.uname().nodename.lower()}"

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

    @property
    def working_dir(self):
        if self._working_dir is None:
            if SP_DEBUG:
                self._working_dir = self._log_dir
                pathlib.Path(self._working_dir).mkdir(parents=True, exist_ok=True)
            else:
                self._working_dir = tempfile.TemporaryDirectory()
        return self._working_dir

    def finish(self):
        if self._working_dir != self._log_dir and SP_DEBUG:
            shutil.copytree(self._working_dir, self._log_dir, dirs_exist_ok=True)
            self._working_dir = None
