import collections
import pathlib
from enum import Flag, auto
from functools import cached_property, reduce
from typing import (Any, Callable, Generic, Iterator, Mapping, MutableMapping,
                    MutableSequence, Optional, Protocol, Sequence, Tuple, Type,
                    TypeVar, Union)

from ..common.SpObject import create_object

from ..plugins.data.file import association as file_association
from ..util.logger import logger
from .Connection import Connection
from .Entry import Entry

_TFile = TypeVar('_TFile', bound='File')


class FileHandler(object):
    def __init__(self, metadata: Mapping, *args, **kwargs) -> None:
        super().__init__()
        self._metadata: Mapping = metadata

    def read(self, lazy=False) -> Entry:
        raise NotImplementedError()

    def write(self, data, lazy=False) -> None:
        raise NotImplementedError()
        


class File(Connection):
    """
        File like object
    """
    class Mode(Flag):
        r = auto()  # open for reading (default)
        w = auto()  # open for writing, truncating the file first
        x = auto()  # open for exclusive creation, failing if the file already exists
        a = auto()  # open for writing, appending to the end of the file if it exists

    def __init__(self,   metadata=None, /, mode="r", **kwargs):

        if isinstance(metadata, pathlib.PosixPath):
            metadata = {"path": metadata}

        if isinstance(mode, str):
            mode = reduce(lambda a, b: a | b, [File.Mode[a] for a in mode])
        elif not isinstance(mode, File.Mode):
            raise TypeError(mode)

        super().__init__(metadata, mode=mode, **kwargs)

        self._holder: FileHandler = None

    def __repr__(self):
        return f"<{self.__class__.__name__} path={self.path}>"

    @property
    def is_valid(self) -> bool:
        return self._holder is not None

    @property
    def is_open(self) -> bool:
        return self._holder is not None

    @property
    def mode(self) -> Mode:
        return self._metadata.get("mode", File.Mode.r)

    @property
    def path(self) -> Union[str, pathlib.Path]:
        return self._metadata.get("path", None)

    def open(self, *args, **kwargs) -> _TFile:

        Connection.open(self)

        protocol = self._metadata.get("protocol", None)

        if protocol in ("http", "https", "ssh"):
            raise NotImplementedError(
                f"TODO: Access to remote files [{protocol}] is not yet implemented!")
        elif protocol not in ("local", "file", None):
            raise NotImplementedError(f"Unsupported protocol {protocol}")

        path = pathlib.Path(self._metadata.get("path", ""))

        if isinstance(path, str):
            path = pathlib.Path(path)
        elif not isinstance(path, pathlib.PosixPath):
            raise RuntimeError(f"Illegal path:{path}")

        path = (pathlib.Path.cwd() / path).expanduser().resolve()

        self._metadata["path"] = path

        file_format = self._metadata.get("format", None)

        if not file_format:
            if not path.suffix:
                raise ValueError(
                    f"Can not guess file format from path! {path}")
            file_format = path.suffix[1:]

        file_class = file_association.get(file_format.lower(), None)

        self._holder = create_object(file_class, self._metadata,
                                     *args, **kwargs)

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


__SP_EXPORT__ = File
