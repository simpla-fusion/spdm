import abc
import datetime
import getpass

from ..utils.Pluggable import Pluggable


class View(Pluggable):
    """Abstract class for all views"""
    _plugin_registry = {}

    def __init__(self, *args, schema="html", **kwargs) -> None:
        self._schema = "html"

    @property
    def schema(self): return self._schema

    @property
    def signature(self) -> str:
        return f"author: {getpass.getuser().capitalize()}. Create by SpDM at {datetime.datetime.now().isoformat()}."

    @abc.abstractmethod
    def display(self, *objs,  **kwargs):
        raise NotImplementedError(f"show() not implemented in {self.__class__.__name__}")


_default_view: View | str = "matplotlib"


def display(obj, schema=None):
    """Show an object"""
    if isinstance(_default_view, str):
        _default_view = View(_default_view)
    return _default_view.display(obj, schema=schema)
