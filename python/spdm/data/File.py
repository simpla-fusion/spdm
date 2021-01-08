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

    @classmethod
    def __new__(cls, data=None,  *args, _metadata=None, format_hint=None,  **kwargs):
        if cls is not File:
            return object.__new__(cls)

        _metadata = collections.ChainMap(_metadata or {}, kwargs)

        if "$class" not in _metadata:
            file_format = _metadata.get("file_format", None) or pathlib.Path(_metadata.get("path", "")).suffix
            file_format = File.extension.get(file_format.lower(), format_hint)
            file_format = file_format.replace('/', '.')

            if not file_format.startswith("file."):
                file_format = "file."+file_format
            _metadata["$class"] = file_format

        logger.debug(_metadata)

        return Document.__new__(data, *args, _metadata=_metadata, **kwargs)

    def __init__(self,  data=None, *args,  path=None, ** kwargs):
        path = path or self.metadata.path

        if not path:
            path = None
        elif isinstance(path, str):
            path = pathlib.Path(path)
        elif isinstance(path, list):
            path = [pathlib.Path(p) for p in path]

        if isinstance(data, pathlib.PosixPath):
            file_format = File.extension.get(data.suffix.lower(), None)
            if file_format is None or f"file/{file_format.lower()}" == self.metadata["$class"]:
                if path is None:
                    path = pathlib.Path.cwd()/data.name
                shutil.copy(data, path)
                data = None
            else:
                data = File(path=data)

        super().__init__(data, *args, path=path, ** kwargs)

    def __repr__(self):
        return str(self.path)

    def __str__(self):
        return str(self.path)

    @property
    def path(self):
        p = getattr(self, "_path", None) or self.metadata.path
        if isinstance(p, str):
            return pathlib.Path(p).resolve()
        elif isinstance(p, (pathlib.PosixPath, list)):
            return p
        elif not p:
            return None

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
        old_d = self.read()
        old_d.update(d)
        self.write(old_d)


__SP_EXPORT__ = File
