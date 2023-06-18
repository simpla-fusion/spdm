from __future__ import annotations
import collections.abc
import abc
import datetime
import getpass

from ..utils.Pluggable import Pluggable
from ..geometry.GeoObject import GeoObject


class View(Pluggable):
    """Abstract class for all views"""
    _plugin_registry = {}

    @classmethod
    def __dispatch__init__(cls, _view_type, self, *args, **kwargs) -> None:
        if _view_type is None or len(_view_type) == 0:
            _view_type = kwargs.pop("type", "matplotlib")

        if isinstance(_view_type, str):
            _view_type = [_view_type,
                          f"spdm.plugins.views.{_view_type}#{_view_type}",
                          f"spdm.plugins.views.{_view_type}{cls.__name__}#{_view_type}{cls.__name__}",
                          f"spdm.plugins.views.{_view_type.capitalize()}#{_view_type.capitalize()}",
                          f"spdm.plugins.views.{_view_type.capitalize()}{cls.__name__}#{_view_type.capitalize()}{cls.__name__}",
                          f"spdm.plugins.views.{cls.__name__}#{_view_type}"
                          ]

        super().__dispatch__init__(_view_type, self, *args, **kwargs)

    def __init__(self, *args,  **kwargs) -> None:
        if self.__class__ is View:
            return View.__dispatch__init__(kwargs.pop("type", None), self, *args, **kwargs)

        self._schema = kwargs.pop("schema", "html")

    @property
    def schema(self): return self._schema

    @property
    def signature(self) -> str:
        return f"author: {getpass.getuser().capitalize()}. Create by SpDM at {datetime.datetime.now().isoformat()}."

    def display(self, *objs,  **kwargs):
        raise NotImplementedError(f"show() not implemented in {self.__class__.__name__}")

    _DEFALUT: str | View = "SVG"


def display(*args,  **kwargs):
    """Show an object"""
    if len(args) == 1 and isinstance(args[0], collections.abc.Sequence):
        args = args[0]

    objs = []
    for obj in args:
        if isinstance(obj, GeoObject):
            objs.append(obj)
        elif hasattr(obj, "__geometry__"):
            objs.append(obj.__geometry__())

    if isinstance(View._DEFALUT, str):
        View._DEFALUT = View(type=View._DEFALUT)
    return View._DEFALUT.display(objs, **kwargs)
