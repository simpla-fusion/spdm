

import collections.abc
import typing

from ..utils.logger import logger
from ..utils.plugin import Pluggable
from .Entry import open_entry
from .Path import Path
from .sp_property import SpDict


class Actor(SpDict, Pluggable):
    mpi_enabled = False

    # _plugin_module_path = "spdm.plugins.actor.actor_{name}"
    _plugin_registry = {}
    # @classmethod
    # def __dispatch_init__(cls, cls_name_list, self,  d=None, *args, default_plugin: str = None,  **kwargs) -> None:
    #     # if isinstance(d, str):
    #     #     d = open_entry(d)
    #     # if plugin_list is None:
    #     #     module_name = None
    #     #     name_path = Path(self.__class__._plugin_name_path)
    #     #     module_name = name_path.fetch(kwargs)
    #     #     if not isinstance(module_name, str) and d is not None:
    #     #         module_name = name_path.fetch(d)
    #     #     if not isinstance(module_name, str):
    #     #         module_name = default_plugin
    #     #     if isinstance(module_name, str):
    #     #         prefix = getattr(self.__class__, "_plugin_prefix", "")
    #     #         if prefix.endswith("/"):
    #     #             module_preifx = self.__class__.__name__.lower()
    #     #             if module_preifx.startswith('_t_'):
    #     #                 module_preifx = module_preifx[3:]
    #     #             prefix += module_preifx
    #     #         if prefix != "" and not prefix.endswith("/"):
    #     #             prefix = prefix+"/"
    #     #         plugin_list = [f"{prefix}{module_name}"]
    #     if cls_name_list is None:
    #         cls_name_list = []
    #     if default_plugin is not None:
    #         cls_name_list.append(default_plugin)
    #     if cls_name_list is None or len(cls_name_list) == 0:
    #         return super().__init__(self, d, *args, **kwargs)
    #     else:
    #         return super().__dispatch_init__(cls_name_list, self, d, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        if self.__class__ is Actor or "_plugin_config" in vars(self.__class__):
            self.__class__.__dispatch_init__(None, self, *args, **kwargs)
            return
        super().__init__(*args, **kwargs)

    def advance(self,  *args, time: float, ** kwargs) -> None:
        logger.debug(f"Advancing {self.__class__.__name__} time={time}")

    def refresh(self,  *args,  ** kwargs) -> None:
        logger.debug(f"Refreshing {self.__class__.__name__} time={getattr(self, 'time', 0.0)}")
