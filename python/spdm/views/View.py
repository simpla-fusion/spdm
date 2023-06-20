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
from ..utils.dict_util import deep_merge_dict


class View(Pluggable):
    """Abstract class for all views"""
    _plugin_registry = {}

    backend = None

    @classmethod
    def __dispatch__init__(cls, _backend_type, self, *args, **kwargs) -> None:

        if isinstance(_backend_type, str):
            _backend_type = [_backend_type,
                             f"spdm.plugins.views.{_backend_type}#{_backend_type}",
                             f"spdm.plugins.views.{_backend_type}{cls.__name__}#{_backend_type}{cls.__name__}",
                             f"spdm.plugins.views.{_backend_type.capitalize()}#{_backend_type.capitalize()}",
                             f"spdm.plugins.views.{_backend_type.capitalize()}{cls.__name__}#{_backend_type.capitalize()}{cls.__name__}",
                             f"spdm.plugins.views.{cls.__name__}#{_backend_type}"
                             ]

        super().__dispatch__init__(_backend_type, self, *args, **kwargs)

    def __init__(self, *args,  **kwargs) -> None:
        if self.__class__ is View:
            return View.__dispatch__init__(kwargs.pop("type", None), self, *args, **kwargs)

        self._styles = kwargs.pop("styles", {})

    @property
    def signature(self) -> str:
        return f"author: {getpass.getuser().capitalize()}. Create by SpDM at {datetime.datetime.now().isoformat()}."

    def display(self, *args,  **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.display")

    def _draw(self, canvas, *args,  **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.draw")

    @typing.final
    def draw(self, canvas, obj, styles, **kwargs):
        if styles is False:
            return

        if not isinstance(styles, collections.abc.Mapping):
            if styles is not None:
                logger.warning(f"ignore unsupported styles {styles}")
            styles = kwargs
        else:
            styles = deep_merge_dict(styles, kwargs)

        if isinstance(obj, str):
            raise NotImplementedError(f"Unsupport type {obj}")

        elif isinstance(obj, tuple):
            o, s = obj
            if not isinstance(s, collections.abc.Mapping):
                logger.warning(f"ignore unsupported styles {s}")

            self.draw(canvas, o, deep_merge_dict(styles, s))

        elif hasattr(obj, "__geometry__"):
            self.draw(canvas, obj.__geometry__, styles)

        elif isinstance(obj, collections.abc.Mapping):
            for k, o in obj.items():
                self.draw(canvas, o, styles.get(k, {}), name=k)
            self.draw(canvas, None, styles)

        elif isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
            for idx, o in enumerate(obj):
                self.draw(canvas, o, styles, index=idx)
            self.draw(canvas, None, styles)

        else:
            self._draw(canvas, obj, styles)

    def profiles(self, *args, **kwargs):
        return self.display(*args, as_profiles=True, **kwargs)

    def draw_profile(self, profile, x, axis=None, ** kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.draw")


_view_instances = {}


def display(*args, output="svg", backend="matplotlib", **kwargs):
    """Show an object"""
    instance = _view_instances.get(backend, None)

    if instance is None:
        instance = _view_instances[backend] = View(type=backend)

    return instance.display(*args, output=output, **kwargs)
