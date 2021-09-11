import collections.abc
from copy import deepcopy
from enum import Flag, auto
from typing import Mapping, TypeVar, Union
import pathlib
from spdm.util.logger import logger
from spdm.util.urilib import urisplit_as_dict

from ..common.SpObject import SpObject

_TConnection = TypeVar('_TConnection', bound='Connection')


class Connection(SpObject):
    class Mode(Flag):
        r = auto()  # open for reading (default)
        w = auto()  # open for writing, truncating the file first
        x = auto()  # open for exclusive creation, failing if the file already exists
        a = auto()  # open for writing, appending to the end of the file if it exists

    def __new__(cls, uri, *args, **kwargs):
        if isinstance(uri, str):
            metadata = urisplit_as_dict(uri)
        elif isinstance(uri, collections.abc.Mapping):
            metadata = deepcopy(uri)
        elif isinstance(uri, (collections.abc.Sequence, pathlib.Path)):
            metadata = {"path": uri}

        return SpObject.create(metadata, uri, *args, **kwargs)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __del__(self):
        if self.is_valid:
            self.close()

    @property
    def is_valid(self) -> bool:
        return False

    def open(self) -> _TConnection:
        # logger.debug(f"[{self.__class__.__name__}]: {self._metadata}")
        return self

    def close(self) -> None:
        # logger.debug(f"[{self.__class__.__name__}]: {self._metadata}")
        return

    def __enter__(self) -> _TConnection:
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
