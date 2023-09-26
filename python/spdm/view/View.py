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
from ..utils.logger import logger, SP_DEBUG
from ..utils.plugin import Pluggable


class View(Pluggable):
    """Abstract class for all views"""

    _plugin_registry = {}
    _plugin_prefix = "spdm.view.view_"

    backend = None

    # @classmethod
    # def __dispatch_init__(cls, _backend_type, self, *args, **kwargs) -> None:

    #     if isinstance(_backend_type, str):
    #         _backend_type = [_backend_type,
    #                          f"spdm.view.{_backend_type}#{_backend_type}",
    #                          f"spdm.view.{_backend_type}{cls.__name__}#{_backend_type}{cls.__name__}",
    #                          f"spdm.view.{_backend_type.capitalize()}#{_backend_type.capitalize()}",
    #                          f"spdm.view.{_backend_type.capitalize()}{cls.__name__}#{_backend_type.capitalize()}{cls.__name__}",
    #                          f"spdm.view.{cls.__name__}#{_backend_type}"
    #                          f"spdm.plugins.view.{_backend_type}#{_backend_type}",
    #                          f"spdm.plugins.view.{_backend_type}{cls.__name__}#{_backend_type}{cls.__name__}",
    #                          f"spdm.plugins.view.{_backend_type.capitalize()}#{_backend_type.capitalize()}",
    #                          f"spdm.plugins.view.{_backend_type.capitalize()}{cls.__name__}#{_backend_type.capitalize()}{cls.__name__}",
    #                          f"spdm.plugins.view.{cls.__name__}#{_backend_type}"
    #                          ]

    #     super().__dispatch_init__(_backend_type, self, *args, **kwargs)

    def __init__(self, *args, **kwargs) -> None:
        if self.__class__ is View:
            View.__dispatch_init__(kwargs.pop("type", None), self, *args, **kwargs)
            return

    @property
    def signature(self) -> str:
        return f"author: {getpass.getuser().capitalize()}. Create by SpDM at {datetime.datetime.now().isoformat()}."

    def render(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.display")

    def profiles(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.draw")


_view_instances = {}


def viewer(backend=None):
    """Get a viewer instance"""

    if backend is None:
        backend = SP_VIEW_BACKEND  # "matplotlib"

    instance = _view_instances.get(backend, None)

    if instance is None:
        instance = _view_instances[backend] = View(type=backend)

    return instance


SP_VIEW_BACKEND = "matplotlib"


def display(*args,   backend=None,  **kwargs):
    """Show an object"""
    return viewer(backend).render(*args,    **kwargs)


def draw_profiles(*args,   backend=None, **kwargs):
    """Show an object"""
    return viewer(backend=backend).profiles(*args,  **kwargs)
