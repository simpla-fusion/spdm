import collections
from copy import deepcopy
import functools
import pathlib
from functools import cached_property, reduce
from typing import (Any, Callable, Generic, Iterator, Mapping, MutableMapping,
                    MutableSequence, Optional, Protocol, Sequence, Tuple, Type,
                    TypeVar, Union)

from spdm.util.urilib import urisplit_as_dict

from ..common.SpObject import SpObject

from ..util.logger import logger
from .Connection import Connection
from .Entry import Entry

from ..plugins.data import file as file_plugins

_TFile = TypeVar('_TFile', bound='File')


class File(Connection):
    """
        File like object
    """

    def __new__(cls, path, *args, format=None, mode="r", **kwargs):
        if cls is not File:
            return SpObject.__new__(cls)

        if isinstance(path, collections.abc.Mapping):
            metadata = deepcopy(path)
        elif isinstance(path, str):
            metadata = urisplit_as_dict(path)
        elif isinstance(path, (list, pathlib.PosixPath)):
            metadata = {"path": path}

        if metadata.get("protocol", None) is None:
            metadata["protocol"] = "file"

        if mode is not None:
            metadata["mode"] = mode

        cls_name = metadata.get("$class", None)

        if cls_name is None:
            format = metadata.get("format", format)
            if format is None:
                path = metadata.get("path", "")
                if isinstance(path, str):
                    format = pathlib.Path(path).suffix[1:]
                elif isinstance(path, pathlib.PosixPath):
                    format = path.suffix[1:]
                else:
                    format = "text"
            cls_name = f".data.file.{format}"
        metadata["$class"] = cls_name

        return SpObject.new_object(metadata)

    def __init__(self,  *args,   **kwargs):
        super().__init__(*args, **kwargs)

        logger.debug(f"Create {self.__class__.__name__}: {self.path}")

        protocol = self._metadata.get("protocol", None)

        if protocol in ("local",  None):
            self._metadata["protocol"] = "file"
        elif protocol in ("http", "https", "ssh"):
            raise NotImplementedError(
                f"TODO: Access to remote files [{protocol}] is not yet implemented!")
        elif protocol != "file":
            raise NotImplementedError(f"Unsupported protocol {protocol}")

    def __repr__(self):
        return f"<{self.__class__.__name__} path={self.path}>"

    @property
    def is_valid(self) -> bool:
        return getattr(self, "_holder", None) is not None

    @property
    def is_open(self) -> bool:
        return getattr(self, "_holder", None) is not None

    @property
    def mode(self) -> Connection.Mode:
        return self._metadata.get("mode", File.Mode.r)

    @property
    def path(self) -> Union[str, pathlib.Path]:
        return self._metadata.get("path", None)

    def open(self, *args, **kwargs) -> _TFile:
        Connection.open(self)
        self._holder = SpObject.create(self._metadata, *args, **kwargs)
        return self

    def close(self):
        self._holder = None
        Connection.close(self)

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

    @property
    def path(self):
        return self._metadata.get("path", "")

    @property
    def mode(self):
        mode = self._metadata.get("mode", "r")
        if isinstance(mode, str):
            mode = functools.reduce(lambda a, b: a | b, [
                                    File.Mode.__members__[a] for a in mode])
        elif not isinstance(mode, File.Mode):
            raise TypeError(mode)
        return mode

    @property
    def mode_str(self):
        mode = self.mode
        return ''.join([(m.name) for m in list(File.Mode) if m & mode])

    def read(self, lazy=False) -> Entry:
        raise NotImplementedError()

    def write(self, data, lazy=False) -> None:
        raise NotImplementedError()


__SP_EXPORT__ = File
