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

    extension = {
        ".bin": "Binary",
        ".h5": "HDF5",
        ".hdf5": "HDF5",
        ".nc": "netCDF",
        ".netcdf": "netCDF",
        ".namelist": "namelist",
        ".nml": "namelist",
        ".xml": "XML",
        ".json": "JSON",
        ".yaml": "YAML",
        ".txt": "TXT",
        ".csv": "CSV",
        ".numpy": "NumPy",
        ".geqdsk": "GEQdsk",
        ".gfile": "GEQdsk",
        ".mds": "mdsplus",
        ".mdsplus": "mdsplus",
    }

    def __new__(cls, _metadata=None, *args, path=None,   file_format=None,  **kwargs):
        if cls is not File and _metadata is None:
            return Document.__new__(cls)
        extension_name = ''
        if not isinstance(_metadata, collections.abc.Mapping) or "$class" not in _metadata:
            file_format = file_format or _metadata.get("file_format", None) or\
                pathlib.Path(path or _metadata.get("path", "")).suffix
            if file_format[0] != '.':
                file_format = '.'+file_format
            extension_name = file_format

            file_format = File.extension.get(file_format.lower(), file_format)
            file_format = file_format.replace('/', '.')

            if not file_format.startswith("file."):
                file_format = "file."+file_format

            if _metadata is None:
                _metadata = file_format
            else:
                _metadata["$class"] = file_format
                _metadata["extension_name"] = extension_name

        return Document.__new__(cls, _metadata=_metadata, *args,  **kwargs)

    def __init__(self,   *args, ** kwargs):

        super().__init__(*args,   ** kwargs)
        if self.path.is_dir():
            self._path /= f"{uuid.uuid1()}{self.metadata.extension_name or '' }"

    def __repr__(self):
        return str(self.path)

    def __str__(self):
        return str(self.path)

    @property
    def path(self):
        return self._path

    @property
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

    @property
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

    @contextlib.contextmanager
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
