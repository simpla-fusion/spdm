import abc
import datetime
import getpass

from ..utils.Pluggable import Pluggable


class View(Pluggable):
    """Abstract class for all views"""
    _plugin_registry = {}

    @classmethod
    def __dispatch__init__(cls, _view_type, self, *args, **kwargs) -> None:
        if _view_type is None or len(_view_type) == 0:
            _view_type = kwargs.pop("type", "matplotlib")

        if isinstance(_view_type, str):
            _view_type = [_view_type,
                         f"spdm.views.{_view_type}#{_view_type}",
                         f"spdm.views.{_view_type}{cls.__name__}#{_view_type}{cls.__name__}",
                         f"spdm.views.{_view_type.capitalize()}#{_view_type.capitalize()}",
                         f"spdm.views.{_view_type.capitalize()}{cls.__name__}#{_view_type.capitalize()}{cls.__name__}",
                         f"spdm.views.{cls.__name__}#{_view_type}"
                         ]

        super().__dispatch__init__(_view_type, self, *args, **kwargs)

    def __init__(self, *args,  **kwargs) -> None:
        if self.__class__ is View:
            return View.__dispatch__init__(None, self, *args, **kwargs)

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
