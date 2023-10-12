from ..utils.logger import logger
from ..utils.plugin import Pluggable
from .sp_property import SpTree


class Actor(SpTree, Pluggable):
    mpi_enabled = False

    _plugin_registry = {}

    def __init__(self, *args, **kwargs):
        if self.__class__ is Actor or "_plugin_prefix" in vars(self.__class__):
            self.__class__.__dispatch_init__(None, self, *args, **kwargs)
            return
        super().__init__(*args, **kwargs)

    def advance(self,  *args, time: float, ** kwargs) -> None:
        logger.debug(f"Advancing {self.__class__.__name__} time={time}")

    def refresh(self,  *args,  ** kwargs) -> None:
        logger.debug(f"Refreshing {self.__class__.__name__} time={getattr(self, 'time', 0.0)}")
