import contextlib
import io
import pathlib
import shutil
import tempfile
import uuid
from typing import TypeVar, Union

from ..utils.logger import logger
from ..utils.uri_utils import uri_split
from .Connection import Connection

_TDirectory = TypeVar('_TDirectory', bound='Directory')


class Directory(Connection):
    """ 
        Default entry for Directory
    """

    def __init__(self, *args, mask=0o777, create_parents=False, **kwargs):
        super().__init__(*args, **kwargs)

        self._mask = mask
        self._create_parents = create_parents

    def __del__(self):
        if self.is_temporary:
            # TODO:
            #   - delete created parents ?
            #   - clear children
            self._dir_path.rmdir

        return super().__del__()

    @property
    def path(self) -> pathlib.Path:
        return self.uri.path

 
    @property
    def cwd(self) -> pathlib.Path:
        if self.path.is_dir():
            pass
        elif self.is_writable:
            self.path.mkdir(mode=self._mask,
                                 parents=self._create_parents and self.is_creatable,
                                 exist_ok=self.is_creatable)  # ??? logical correct?
        else:
            raise NotADirectoryError(self.path)
        return self.path

    def cd(self, path) -> _TDirectory:
        return self.__class__(self.path/path, mask=self._mask, create_parents=self._create_parents, mode=self.mode)
