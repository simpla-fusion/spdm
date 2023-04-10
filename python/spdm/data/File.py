from __future__ import annotations

import collections
import collections.abc
import pathlib
import typing
from ..util.uri_utils import URITuple, uri_split
from ..util.sp_export import sp_load_module
from .Connection import Connection
from .Entry import Entry


class File(Connection):
    """
        File like object
    """
    _registry = {}
    _plugin_prefix = "spdm.plugins.data.Plugin"

    # @classmethod
    # def register(cls, name: typing.Union[str, typing.List[str]], other_cls=None):
    #     """
    #     Decorator to register a class to the registry.
    #     """
    #     if other_cls is not None:
    #         if isinstance(name, str):
    #             cls._registry[name] = other_cls
    #         elif isinstance(name, collections.abc.Sequence):
    #             for n in name:
    #                 cls._registry[n] = other_cls

    #         return other_cls
    #     else:
    #         def decorator(o_cls):
    #             File.register(name, o_cls)
    #             return o_cls
    #         return decorator

    @classmethod
    def create(cls, path, *args, **kwargs):
        if cls is not File:
            return cls(cls, path, *args, **kwargs)

        n_cls_name = '.'
        if "format" in kwargs:
            n_cls_name = kwargs.get("format")
        elif isinstance(path, collections.abc.Mapping):
            n_cls_name = path.get("$class", None)
        elif isinstance(path,   pathlib.PosixPath):
            n_cls_name = path.suffix[1:].upper()
        elif isinstance(path, (str, URITuple)):
            uri = uri_split(path)
            if isinstance(uri.format, str):
                n_cls_name = uri.format
            else:
                n_cls_name = pathlib.PosixPath(uri.path).suffix[1:].upper()
        if n_cls_name == ".":
            n_cls_name = ".text"

        return super().create(n_cls_name, path, *args, **kwargs)
        # n_cls = cls._registry.get(n_cls_name, None)
        # if n_cls is None:
        #     n_cls = sp_load_module(f"spdm.plugins.data.Plugin{n_cls_name}#{n_cls_name}File")
        # if n_cls is not None:
        #     return n_cls(path, *args, **kwargs)
        # else:
        #     raise NotImplementedError(f"Cannot create file for {path}")

    def __init__(self,  *args, mode="r", ** kwargs):
        """
         r       Readonly, file must exist (default)
         rw      Read/write, file must exist
         w       Create file, truncate if exists
         x       Create file, fail if exists
         a       Read/write if exists, create otherwise
        """
        if isinstance(mode, str):
            mode = File.INV_MOD_MAP.get(mode, File.Mode.read)
        super().__init__(*args, mode=mode, **kwargs)

    @property
    def mode_str(self) -> str:
        return File.MOD_MAP.get(self.mode, "r")

    @property
    def entry(self) -> Entry:
        if self.is_readable:
            return self.read()
        else:
            return self.write()

    def read(self, lazy=False) -> Entry:
        if self._holder is None:
            self.open()
        return self._holder.read(lazy=lazy)

    def write(self, *args, **kwargs):
        if not self.is_open:
            self.open()
        self._holder.write(*args, **kwargs)

    def __enter__(self) -> File:
        return super().__enter__()

    def read(self, lazy=False) -> Entry:
        raise NotImplementedError()

    def write(self, data, lazy=False) -> Entry:
        raise NotImplementedError()


__SP_EXPORT__ = File
