

from ..utils.plugin import Pluggable
from ..utils.logger import logger
from .sp_property import SpDict


class Actor(SpDict, Pluggable):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def advance(self,  *args, time: float, ** kwargs) -> None:
        logger.debug(f"Advancing {self.__class__.__name__} time={time}")

    def refresh(self,  *args,  ** kwargs) -> None:
        logger.debug(f"Refreshing {self.__class__.__name__} time={getattr(self, 'time', 0.0)}")
