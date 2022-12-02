import collections

import pathlib
from copy import deepcopy
from typing import TypeVar, Union

from ..common.tags import _undefined_
from ..plugins.data import file as file_plugins
from ..util.logger import logger
from ..util.uri_utils import URITuple, uri_split
from .Connection import Connection
from .Entry import Entry
from .SpObject import SpObject

_TFile = TypeVar('_TFile', bound='File')


class File(Connection):
    """
        File like object
    """

    def __new__(cls, path, *args, **kwargs):
        if cls is not File:
            return object.__new__(cls)

        n_cls_name = '.'
        if "format" in kwargs:
            format = kwargs.get("format")
            n_cls_name = f".{format.lower()}"
        elif isinstance(path, collections.abc.Mapping):
            n_cls_name = path.get("$class", None)
        elif isinstance(path,   pathlib.PosixPath):
            n_cls_name = path.suffix.lower()
        elif isinstance(path, (str, URITuple)):
            uri = uri_split(path)
            if isinstance(uri.format, str):
                n_cls_name = f".{uri.format.lower()}"
            else:
                n_cls_name = pathlib.PosixPath(uri.path).suffix.lower()
        if n_cls_name == ".":
            n_cls_name = ".text"
        return File.object_new(n_cls_name)

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def is_valid(self) -> bool:
        return getattr(self, "_holder", None) is not None

    @property
    def is_open(self) -> bool:
        return getattr(self, "_holder", None) is not None

    def open(self, *args, **kwargs) -> _TFile:
        super().open()
        return self

    def close(self):

        super().close()

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

    def __enter__(self) -> _TFile:
        return super().__enter__()

    def read(self, lazy=False) -> Entry:
        raise NotImplementedError()

    def write(self, data, lazy=False) -> Entry:
        raise NotImplementedError()


__SP_EXPORT__ = File
