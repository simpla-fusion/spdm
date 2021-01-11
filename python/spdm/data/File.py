import collections
import contextlib
import inspect
import io
import pathlib
import shutil
import tempfile
import uuid

from ..util.logger import logger
from ..util.sp_export import sp_find_module
from ..util.urilib import urisplit
from .DataObject import DataObject
from .Document import Document


class File(Document):
    """ Default entry for file-like object
    """

    def __new__(cls, _metadata=None, *args, path=None,    file_format=None,  **kwargs):
        if cls is not File and _metadata is None:
            return Document.__new__(cls)
        if path is None and isinstance(_metadata, str):
            path = _metadata
            _metadata = None
        if isinstance(path, (str, pathlib.PosixPath)):
            extension_name = pathlib.Path(path or _metadata.get("path", "")).suffix
        else:
            extension_name = ""

        if not isinstance(_metadata, collections.abc.Mapping) or "$class" not in _metadata:

            class_name = file_format or _metadata.get("file_format", None)

            if not class_name and not not extension_name:
                class_name = extension_name[1:]

            if not class_name.startswith("."):
                class_name = "file."+file_format

            if _metadata is None:
                _metadata = class_name
            else:
                _metadata["$class"] = class_name
                _metadata["extension_name"] = extension_name

        return Document.__new__(cls, _metadata=_metadata, *args,  **kwargs)

    def __init__(self,  _metadata=None, *args, path=None, ** kwargs):
        if path is None and isinstance(_metadata, str):
            path = _metadata
            _metadata = None
        super().__init__(_metadata=_metadata, *args,  path=path, ** kwargs)

        self._path = self._path or self.metadata.path or "."
        if not isinstance(self._path, list):
            self._path = pathlib.Path.cwd() / self._path
            if self.path.is_dir():
                self._path /= f"{uuid.uuid1()}{self.extension_name or '.txt' }"
            self._path = self._path.expanduser().resolve()

    @ property
    def extension_name(self):
        return self.metadata.extension_name or '.txt'

    def __repr__(self):
        return str(self.path)

    def __str__(self):
        return str(self.path)

    @ property
    def template(self):
        p = getattr(self, "_template", None) or self.metadata.template
        if isinstance(p, str):
            return pathlib.Path(p).resolve()
        elif isinstance(p, pathlib.PosixPath):
            return p
        elif not p:
            return None
        else:
            raise ValueError(p)

    @ property
    def is_writable(self):
        return "w" in self.mode or "x" in self.mode

    def flush(self, *args, **kwargs):
        pass

    def copy(self, path=None):
        if path is None:
            path = f"{self._path.stem}_copy{self._path.suffix}"
        elif isinstance(path, str):
            path = pathlib.Path(path)
            if path.is_dir() and path != self._path.parent:
                path = path/self._path.name
        shutil.copy(self._path, path.as_posix())
        res = self.__class__(path)
        res._mode = self._mode
        res._buffering = self._buffering
        res._encoding = self._encoding
        res._errors = self._errors
        res._newline = self._newline
        res._schema = self._schema
        return res

    @ contextlib.contextmanager
    def open(self, mode=None, buffering=None, encoding=None, newline=None):
        if isinstance(self._path, pathlib.Path):
            path = self._path
        else:
            o = urisplit(self._path)
            path = pathlib.Path(o.path)

        fid = path.open(mode=mode or self._mode)

        yield fid

        if fid is not None:
            fid.close()

    def read(self, *args, **kwargs):
        with self.open(mode="r") as fid:
            res = fid.read(*args, **kwargs)
        return res

    def write(self, d, *args, **kwargs):
        with self.open() as fid:
            fid.write(d, *args, **kwargs)

    def update(self, d, *args, **kwargs):
        if d is None:
            return
        elif isinstance(d, pathlib.PosixPath):
            self._path = d
        # old_d = self.read()
        # old_d.update(d)
        # self.write(old_d)


__SP_EXPORT__ = File
