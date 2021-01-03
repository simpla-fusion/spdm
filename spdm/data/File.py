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

    associations = {
        "file.general": ".data.file.GeneralFile",
        "file.bin": ".data.file.Binary",
        "file.h5": ".data.file.HDF5",
        "file.hdf5": ".data.file.HDF5",
        "file.nc": ".data.file.netCDF",
        "file.netcdf": ".data.file.netCDF",
        "file.namelist": ".data.file.namelist",
        "file.nml": ".data.file.namelist",
        "file.xml": ".data.file.XML",
        "file.json": ".data.file.JSON",
        "file.yaml": ".data.file.YAML",
        "file.txt": ".data.file.TXT",
        "file.csv": ".data.file.CSV",
        "file.numpy": ".data.file.NumPy",
        "file.geqdsk": ".data.file.GEQdsk",
        "file.gfile": ".data.file.GEQdsk",
        "file.mds": ".data.db.MDSplus#MDSplusDocument",
        "file.mdsplus": ".data.db.MDSplus#MDSplusDocument",
    }

    @staticmethod
    def __new__(cls, data=None,  *args, _metadata=None, file_format=None, **kwargs):
        if cls is not File and _metadata is None:
            return object.__new__(cls)

        if file_format is not None:
            pass
        elif isinstance(_metadata, str):
            file_format = _metadata
        elif isinstance(_metadata, collections.abc.Mapping):
            file_format = _metadata.get("$class", None) or _metadata.get("$schema", None)
            # raise TypeError(f"{type(_metadata)} is not a 'dict'!")

        if file_format is None and isinstance(data, str):
            o = urisplit(data)

            if o.schema in (None, 'local', 'file'):
                file_format = pathlib.Path(o.path).suffix

        if not isinstance(file_format, str):
            raise ValueError(f"File format is not defined! {file_format}")

        file_format = file_format.replace('/', '.')

        if file_format.startswith("."):
            file_format = "file"+file_format
        logger.debug(file_format)
        n_cls = File.associations.get(file_format, file_format)

        _metadata = collections.ChainMap({"$class": n_cls}, _metadata or {})

        return Document.__new__(data, *args, _metadata=_metadata, **kwargs)

    def __init__(self,  data=None, *args,  path=None,   ** kwargs):
        super().__init__(*args,   ** kwargs)

        if path is None:
            self._path = pathlib.Path.cwd()
        elif isinstance(path, str):
            self._path = pathlib.Path.cwd()/path
        elif isinstance(path, collections.abc.Sequence):
            self._path = [pathlib.Path.cwd()/p for p in path]
        else:
            raise TypeError(type(path))

    def __repr__(self):
        return str(self._path)

    def __str__(self):
        return str(self._path)

    @property
    def path(self):
        return self._path

    @property
    def is_writable(self):
        return "w" in self.metadata.mode or "x" in self.metadata.mode

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
