from ..view import View as sp_view
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

    def _repr_svg_(self) -> str:
        try:
            res = sp_view.display(self, output="svg")
        except Exception as error:
            logger.error(error)
            res = None
        return res

    def __geometry__(self,  *args,  **kwargs):
        return {}
