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

    @staticmethod
    def __new__(cls, data=None,  *args, _metadata=None, path=None, file_format=None, **kwargs):
        if cls is not File and _metadata is None:
            return object.__new__(cls)

        if isinstance(_metadata, collections.abc.Mapping) and "$class" in _metadata:
            pass
        elif file_format is None:
            if path is None and isinstance(data, str):
                path = data
            o = urisplit(path)

            if o.schema in (None, 'local', 'file'):
                ext = pathlib.Path(o.path).suffix
                file_format = File.extension.get(ext.lower(), ext)

        if isinstance(file_format, str):
            # raise ValueError(f"File format is not defined! {file_format}")
            file_format = file_format.replace('/', '.')

            if not file_format.startswith("file."):
                file_format = "file."+file_format

            _metadata = collections.ChainMap({"$class": file_format}, _metadata or {})

        return Document.__new__(data, *args, _metadata=_metadata, **kwargs)

    def __init__(self,  data=None, *args,  path=None, working_dir=None,   ** kwargs):

        if working_dir is None:
            working_dir = pathlib.Path.cwd()
        elif isinstance(working_dir, str):
            working_dir = pathlib.Path.cwd()/path
        elif not isinstance(working_dir, pathlib.PosixPath):
            raise TypeError(type(working_dir))

        if path is None:
            path = str(self.metadata.path)

        if isinstance(path, str):
            path = working_dir/path
        if isinstance(path, list):
            path = [working_dir/p for p in path]

        super().__init__(data, *args, path=path, ** kwargs)

        if data is not None:
            self.update(data)

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
        try:
            fid = path.open(
                mode=mode or self._mode,
                buffering=buffering or self._buffering,
                encoding=encoding or self._encoding,
                newline=newline or self._newline)
        except Exception:
            fid = None
        finally:
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
