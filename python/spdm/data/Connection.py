from __future__ import annotations

import typing
from enum import Flag, auto

from ..utils.Plugin import Pluggable
from ..utils.uri_utils import URITuple, uri_split
from .Entry import Entry


class Connection(Pluggable):
    _registry = {}

    class Mode(Flag):
        read = auto()       # open for reading (default)
        write = auto()      # open for writing, truncating the file first
        create = auto()     # open for exclusive creation, failing if the file already exists
        append = read | write | create
        temporary = auto()  # is temporary
    """
        r       Readonly, file must exist (default)
        r+      Read/write, file must exist
        w       Create file, truncate if exists
        w- or x Create file, fail if exists
        a       Read/write if exists, create otherwise
    """
    MOD_MAP = {Mode.read: "r",
               Mode.read | Mode.write: "rw",
               Mode.write: "x",
               Mode.write | Mode.create: "w",
               Mode.read | Mode.write | Mode.create: "a",
               }
    INV_MOD_MAP = {"r": Mode.read,
                   "rw": Mode.read | Mode.write,
                   "x": Mode.write,
                   "w": Mode.write | Mode.create,
                   "a": Mode.read | Mode.write | Mode.create,
                   }

    class Status(Flag):
        opened = auto()
        closed = auto()

    def __init__(self, uri, *args,  mode=Mode.read, **kwargs):
        """
         r       Readonly, file must exist (default)
         rw      Read/write, file must exist
         w       Create file, truncate if exists
         x       Create file, fail if exists
         a       Read/write if exists, create otherwise
        """
        if self.__class__ is Connection:
            Pluggable.__init__(self, *args, mode=mode, **kwargs)
            return
                
        self._uri = uri_split(uri)
        self._mode = Connection.INV_MOD_MAP[mode] if isinstance(mode, str) else mode
        self._is_open = False

    def __del__(self):
        if self.is_open:
            self.close()

    def __repr__(self):
        return f"<{self.__class__.__name__} path={self.uri.path} protocol={self.uri.protocol} format={self.uri.format}>"

    @property
    def uri(self) -> URITuple:
        return self._uri

    @property
    def path(self) -> typing.Any:
        return self.uri.path

    @property
    def mode(self) -> Mode:
        return self._mode

    # @property
    # def mode_str(self) -> str:
    #     return ''.join([(m.name[0]) for m in list(Connection.Mode) if m & self._mode])

    @property
    def is_readable(self) -> bool:
        return bool(self._mode & Connection.Mode.read)

    @property
    def is_writable(self) -> bool:
        return bool(self._mode & Connection.Mode.write)

    @property
    def is_creatable(self) -> bool:
        return bool(self._mode & Connection.Mode.create)

    @property
    def is_temporary(self) -> bool:
        return bool(self._mode & Connection.Mode.temporary)

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> Connection:
        self._is_open = True
        return self

    def close(self) -> None:
        self._is_open = False
        return

    @property
    def entry(self) -> Entry:
        raise NotImplementedError()

    def __enter__(self) -> Connection:
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
