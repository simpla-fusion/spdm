import contextlib
import io
import pathlib
import shutil
import tempfile
import uuid

from ..util.logger import logger
from ..util.uri_utils import uri_split
from .Connection import Connection
from .Collection import Collection


class Directory(Connection):
    """ 
        Default entry for Directory
    """

    def __init__(self,   *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._dir = mkdir(self._path)

    def __del__(self):
        if self.is_temporary:
            del self._path

        return super().__del__()
