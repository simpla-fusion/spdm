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
from .Document import Document


class File(Document):
    """ Default entry for file-like object
    """
    extensions = {

    }

    file_associations = {
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
    def __new__(cls, data=None,  *args, metadata=None, **kwargs):
        if cls is not File and metadata is None:
            return object.__new__(cls)

        if isinstance(metadata, str):
            metadata = io.read(metadata)
        elif metadata is None:
            metadata = {}
        elif not isinstance(metadata, collections.abc.Mapping):
            raise TypeError(f"{type(metadata)} is not a 'dict'!")

        if isinstance(data, str) and "$class" not in metadata:
            o = urisplit(data)
            if o.schema not in (None, 'local', 'file'):
                raise NotImplementedError("TODO: fetch remote file!")
            else:
                metadata.setdefault("$class", pathlib.Path(o.path).suffix)

        n_cls = metadata.get("$class", ".general")

        if isinstance(n_cls, str):
            if n_cls.startswith("."):
                n_cls = "file"+n_cls
            n_cls = File.associations.get(n_cls, n_cls)

            metadata = collections.ChainMap({"$class": n_cls}, metadata or {})

        return Document.__new__(data, *args, metadata=metadata, **kwargs)

    def __init__(self,  data=None, *args,   working_dir=None,   ** kwargs):
        super().__init__(data, *args,   working_dir=working_dir, ** kwargs)

        if working_dir is None:
            working_dir = pathlib.Path.cwd()
        else:
            working_dir = pathlib.Path(working_dir)

        file_path = self.metadata.path or self.metadata.schema.path

        if not file_path:
            file_path = ""

        self._path = working_dir/str(file_path)

    def __repr__(self):
        return str(self._path)

    def __str__(self):
        return str(self._path)

    @ property
    def path(self):
        return self._path

    @ property
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

    @ contextlib.contextmanager
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
