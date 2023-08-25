from __future__ import annotations

import collections
import collections.abc
import datetime
import getpass
import typing

import matplotlib.pyplot as plt
#  在 MatplotlibView 中 imported matplotlib 会不起作用
#  报错 : AttributeError: module 'matplotlib' has no attribute 'colors'. Did you mean: 'colormaps'?
from ..utils.tree_utils import merge_tree_recursive
from ..utils.logger import logger
from ..utils.plugin import Pluggable


class View(Pluggable):
    """Abstract class for all views"""
    _plugin_registry = {}

    backend = None

    @classmethod
    def __dispatch__init__(cls, _backend_type, self, *args, **kwargs) -> None:

        if isinstance(_backend_type, str):
            _backend_type = [_backend_type,
                             f"spdm.views.{_backend_type}#{_backend_type}",
                             f"spdm.views.{_backend_type}{cls.__name__}#{_backend_type}{cls.__name__}",
                             f"spdm.views.{_backend_type.capitalize()}#{_backend_type.capitalize()}",
                             f"spdm.views.{_backend_type.capitalize()}{cls.__name__}#{_backend_type.capitalize()}{cls.__name__}",
                             f"spdm.views.{cls.__name__}#{_backend_type}"
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

    def render(self, *args,  **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.display")

    def _draw(self, canvas, *args,  **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.draw")

    def draw(self, canvas, obj, styles):
        """Draw an object on canvas"""

        if styles is False:
            return
        elif styles is True:
            styles = {}
        elif not isinstance(styles, collections.abc.Mapping):
            raise TypeError(f"styles must be a dict, not {type(styles)}")

        if isinstance(obj, tuple):
            o, s = obj
            if s is False:
                styles = False
            elif s is True:
                pass
            elif isinstance(s, collections.abc.Mapping):
                styles = merge_tree_recursive(styles, s)
            else:
                logger.warning(f"ignore unsupported styles {s}")

            self.draw(canvas, o, styles)

        elif hasattr(obj.__class__, "__geometry__"):
            self.draw(canvas, obj.__geometry__, styles)

        elif isinstance(obj, dict):
            for k, o in obj.items():
                self.draw(canvas, o, collections.ChainMap({"id": k}, styles.get(k, {})))

            self.draw(canvas, None,  styles)

        elif isinstance(obj, list):
            for idx, o in enumerate(obj):
                self.draw(canvas, o, collections.ChainMap({"id": idx}, styles))

            self.draw(canvas, None, styles)

        else:
            self._draw(canvas, obj, styles)

    def profiles(self, *args, **kwargs):
        return self.render(*args, as_profiles=True, **kwargs)

    def draw_profile(self, profile, x, axis=None, ** kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.draw")


_view_instances = {}


def viewer(output=None, backend=None):
    """Get a viewer instance"""

    if backend is None and isinstance(output, str):
        backend = output.split('.')[-1]

    if backend is None:
        backend = "matplotlib"

    instance = _view_instances.get(backend, None)

    if instance is None:
        instance = _view_instances[backend] = View(type=backend)

    return instance


def display(*args, output=None, backend=None, **kwargs):
    """Show an object"""

    return viewer(backend=backend, output=output).render(*args, output=output, **kwargs)
