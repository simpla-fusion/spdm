from __future__ import annotations

import collections.abc
import pathlib
import typing

from ..utils.logger import logger
from ..utils.tags import _undefined_
from ..utils.uri_utils import URITuple, uri_split
from .Document import Document
from .Entry import Entry


class File(Document):
    """
        File like object
    """

    def __init__(self, url: str | pathlib.Path | URITuple, *args, format=None, default_format=None, **kwargs):
        if self.__class__ is File:
            if format is not None:
                format = format.lower()
            elif isinstance(url, dict):
                format = url.get("$class", "").lower()
            elif isinstance(url,   pathlib.PosixPath):
                format = url.suffix[1:].lower()
            elif isinstance(url, (str, URITuple)):
                uri = uri_split(url)
                schemes = uri.protocol.split("+") if isinstance(uri.protocol, str) else []
                if len(schemes) == 0:
                    pass
                elif schemes[0] in ["file", "local"]:
                    format = "+".join(schemes[1:])

                if format is None or format == "":
                    format = pathlib.PosixPath(uri.path).suffix[1:]

            super().__dispatch__init__([format, default_format], self, url, *args, **kwargs)

            return

        super().__init__(url, *args, **kwargs)
        self._is_open = False

    @property
    def mode_str(self) -> str: return File.MOD_MAP.get(self.mode, "r")

    @property
    def entry(self) -> Entry:
        if self.is_readable:
            return self.read()
        else:
            return self.write()

    def __enter__(self) -> Document:
        return super().__enter__()

    def read(self, lazy=False) -> Entry:
        raise NotImplementedError()

    def write(self, data=None, lazy=False) -> Entry:
        raise NotImplementedError()
