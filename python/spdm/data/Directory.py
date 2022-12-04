import contextlib
import io
import pathlib
import shutil
import tempfile
import uuid

from ..util.logger import logger
from ..util.uri_utils import uri_split
from .Connection import Connection
from typing import TypeVar, Union

_TDirectory = TypeVar('_TDirectory', bound='Directory')


class Directory(Connection):
    """ 
        Default entry for Directory
    """

    def __init__(self, *args, mask=0o777, create_parents=False, **kwargs):
        super().__init__(*args, **kwargs)

        parts = pathlib.Path(self.uri.path).parts
        idx, s = next(filter(lambda s: '{' in s[1] or "*" in s[1], enumerate(parts)))

        self._dir_path = pathlib.Path(*list(parts)[:idx])
        self._glob = "/".join(parts[idx:])
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
        return self._dir_path

    @property
    def glob(self) -> str:
        return self._glob

    @property
    def cwd(self) -> pathlib.Path:
        if self._dir_path.is_dir():
            pass
        elif self.is_writable:
            self._dir_path.mkdir(mode=self._mask,
                                 parents=self._create_parents and self.is_creatable,
                                 exist_ok=self.is_creatable)  # ??? logical correct?
        else:
            raise NotADirectoryError(self._dir_path)
        return self._dir_path

    def cd(self, path) -> _TDirectory:
        return self.__class__(self.path/path, mask=self._mask, create_parents=self._create_parents, mode=self.mode)
