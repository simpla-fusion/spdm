

import collections.abc

from ..utils.logger import logger
from ..utils.plugin import Pluggable
from .Path import Path
from .sp_property import SpDict
from .open_entry import open_entry


class Actor(SpDict, Pluggable):
    mpi_enabled = False

    _plugin_prefix = ""
    _plugin_name_path = "plugin_name"

    @classmethod
    def __dispatch__init__(cls, plugin_list, self,  d=None, *args, default_plugin: str = None,  **kwargs) -> None:
        if isinstance(d, str):
            d = open_entry(d)

        if plugin_list is None:
            module_name = None
            name_path = Path(self.__class__._plugin_name_path)

            module_name = name_path.fetch(kwargs)

            if not isinstance(module_name, str) and d is not None:
                module_name = name_path.fetch(d)

            if not isinstance(module_name, str):
                module_name = default_plugin

            if isinstance(module_name, str):
                prefix = getattr(self.__class__, "_plugin_prefix", "")
                if prefix.endswith("/"):
                    module_preifx = self.__class__.__name__.lower()
                    if module_preifx.startswith('_t_'):
                        module_preifx = module_preifx[3:]
                    prefix += module_preifx
                if prefix != "" and not prefix.endswith("/"):
                    prefix = prefix+"/"
                plugin_list = [f"{prefix}{module_name}"]

        if plugin_list is None or len(plugin_list) == 0:
            return super().__init__(self, d, *args, **kwargs)
        else:
            return super().__dispatch__init__(plugin_list, self, d, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        if self.__class__ is Actor or "_plugin_registry" in vars(self.__class__):
            Actor.__dispatch__init__(None, self, *args, **kwargs)
            return
        super().__init__(*args, **kwargs)

    # def __init__(self, *args, **kwargs) -> None:
    #     super().__init__(*args, **kwargs)
    #     logger.debug(f"{self.__class__.__name__} MPI_ENBLAED={self.mpi_enabled}")

    def advance(self,  *args, time: float, ** kwargs) -> None:
        logger.debug(f"Advancing {self.__class__.__name__} time={time}")

    def refresh(self,  *args,  ** kwargs) -> None:
        logger.debug(f"Refreshing {self.__class__.__name__} time={getattr(self, 'time', 0.0)}")
