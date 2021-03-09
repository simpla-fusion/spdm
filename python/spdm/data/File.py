import collections
import contextlib
from spdm.data.DataObject import DataObject
import inspect
import io
import pathlib
import shutil
import tempfile
import uuid

from ..util.logger import logger
from ..util.urilib import urisplit
from .Document import Document


Document.schema.update(
    {
        "file.table": ".data.file.Table",
        "file.bin": ".data.file.Binary",
        "file.h5": ".data.file.HDF5",
        "file.hdf5": ".data.file.HDF5",
        "file.nc": ".data.file.NetCDF",
        "file.netcdf": ".data.file.NetCDF",
        "file.namelist": ".data.file.namelist",
        "file.nml": ".data.file.namelist",
        "file.xml": ".data.file.XML",
        "file.json":  ".data.file.JSON",
        "file.yaml": ".data.file.YAML",
        "file.txt": ".data.file.TXT",
        "file.csv": ".data.file.CSV",
        "file.numpy": ".data.file.NumPy",
        "file.geqdsk": ".data.file.GEQdsk",
        "file.gfile": ".data.file.GEQdsk",
        "file.mds": ".data.db.MDSplus#MDSplusDocument",
        "file.mdsplus": ".data.db.MDSplus#MDSplusDocument",
        "db.imas": ".data.db.IMAS#IMASDocument",
    }
)


class File(Document):
    """ 
        Default entry for file-like object
    """

    @staticmethod
    def __new__(cls,  metadata=None, *args, **kwargs):

        if metadata is not None and not isinstance(metadata, collections.abc.Mapping):
            metadata = {"path": metadata}

        metadata = collections.ChainMap(metadata or {}, kwargs)

        if cls is not File and len(metadata) == 0:
            return Document.__new__(cls)

        n_cls = metadata.get("$class", None)

        if not n_cls:
            file_format = metadata.get("format", None)

            if not file_format:
                path = pathlib.Path(metadata.get("path", ""))

                if not path.suffix:
                    raise ValueError(f"Can not guess file format from path! {path}")
                file_format = path.suffix[1:]

            n_cls = f"file.{file_format}"

        return Document.__new__(cls, n_cls)

    def __init__(self, path=None,  *args, ** kwargs):
        super().__init__(path, *args,  ** kwargs)
        if not isinstance(self._path, list):
            self._path = pathlib.Path.cwd() / self._path
            if self.path.is_dir():
                self._path /= f"{uuid.uuid1()}{self.extension_name or '.txt' }"
            self._path = self._path.expanduser().resolve()

    @property
    def extension_name(self):
        return self.metadata.get("extension_name", '.txt')

    def __repr__(self):
        return str(self.path)

    def __str__(self):
        return str(self.path)

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
        if isinstance(d, pathlib.PosixPath):
            self._path = d
        else:
            super().update(d)
        # old_d = self.read()
        # old_d.update(d)
        # self.write(old_d)


DataObject.schema['file'] = File

__SP_EXPORT__ = File
