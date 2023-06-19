from __future__ import annotations

import abc
import collections
import collections.abc
import datetime
import getpass
import typing

import matplotlib.pyplot as plt
import numpy as np

from ..utils.logger import logger
from ..utils.Pluggable import Pluggable
from ..utils.typing import array_type


class View(Pluggable):
    """Abstract class for all views"""
    _plugin_registry = {}
    _DEFALUT: str | View = "matplotlib"
    backend = None

    @classmethod
    def __dispatch__init__(cls, _view_type, self, *args, **kwargs) -> None:
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

    def display(self, *args,  **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.display")

    def draw_one(self, obj, canvas=None, styles=None,  **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.draw")

    @typing.final
    def draw(self, obj, canvas=None, styles=None,  **kwargs):
        if styles is False:
            return
        elif isinstance(styles, collections.abc.Mapping) and f"${self.backend}" in styles:
            styles = styles[f"${self.backend}"]

        if isinstance(obj, str):
            raise NotImplementedError(f"Unsupport type {obj}")

        elif isinstance(obj, tuple):
            o, s = obj
            if not isinstance(s, collections.abc.Mapping):
                pass
            elif isinstance(styles, collections.abc.Mapping):
                s = collections.ChainMap(s, styles)
            self.draw(o,  canvas, styles=s)
            
        elif hasattr(obj, "__geometry__"):
            self.draw(obj.__geometry__, canvas, styles=styles, **kwargs)

        elif isinstance(obj, collections.abc.Mapping):
            for k, o in obj.items():
                if k.startswith("$"):
                    continue
                s = styles.get(k, {}) if isinstance(styles, collections.abc.Mapping) else None
                self.draw(o,  canvas, styles=s, name=k)
        elif isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
            for idx, o in enumerate(obj):
                self.draw(o,  canvas, styles=styles, name=getattr(o, "name", str(idx)))

        else:
            self.draw_one(obj,  canvas, styles=styles)

    def profiles(self, *args, **kwargs):
        return self.display(*args, as_profiles=True, **kwargs)

    def draw_profile(self, profile, x, axis=None, ** kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.draw")


def display(*args,  **kwargs):
    """Show an object"""
    # if len(args) == 1 and isinstance(args[0], collections.abc.Sequence):
    #     args = args[0]

    # objs = []
    # for obj in args:
    #     if isinstance(obj, GeoObject):
    #         objs.append(obj)
    #     elif hasattr(obj, "__geometry__"):
    #         objs.append(obj.__geometry__())

    if isinstance(View._DEFALUT, str):
        View._DEFALUT = View(type=View._DEFALUT)

    return View._DEFALUT.display(*args, **kwargs)
