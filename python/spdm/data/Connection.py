import collections
from typing import Mapping, TypeVar, Union
from spdm.util.logger import logger
from spdm.util.urilib import urisplit_as_dict
from ..common.SpObject import SpObject


_TConnection = TypeVar('_TConnection', bound='Connection')


class Connection(SpObject):

    def __init__(self, metadata,  **kwargs) -> None:
        if isinstance(metadata, str):
            metadata = urisplit_as_dict(metadata)
        super().__init__(metadata, **kwargs)

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
