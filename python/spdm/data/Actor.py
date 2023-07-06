

from ..utils.plugin import Pluggable
from ..utils.logger import logger
from .sp_property import SpDict


class Actor(SpDict, Pluggable):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def advance(self,  *args, time: float, ** kwargs) -> float:
        logger.debug(f"Advancing {self.__class__.__name__} time={time}")
        return getattr(self, "time", 0.0)

    def refresh(self,  *args,  ** kwargs) -> float:
        logger.debug(f"Refreshing {self.__class__.__name__}")
        return getattr(self, "time", 0.0)
