import inspect
from spdm.util.sp_export import sp_find_module
import contextlib
import io
import pathlib
import shutil
import tempfile
import uuid
import collections
from ..util.logger import logger
from ..util.urilib import urisplit
from .Document import Document


class File(Document):
    """ Default entry for file-like object
    """
    #     if isinstance(desc, collections.abc.Mapping):
    #         path = desc.get("path", None)
    #         file_format = file_format or desc.get("file_format", None)
    #     elif isinstance(desc, (str, pathlib.PosixPath)):
    #         path = desc
    #     else:
    #         raise TypeError(f"{type(desc)}")

    #     if file_format is not None:
    #         pass
    #     elif isinstance(path, (str, pathlib.PosixPath)):
    #         path = pathlib.Path(path)
    #         file_format = path.suffix[1:]
    #     else:
    #         raise ValueError(f"'file_format' is not defined!")

    #     n_cls = File.associations.get(file_format.lower(), f"{__package__}.file.{file_format}")

    #     if isinstance(n_cls, str):
    #         n_cls = sp_find_module(n_cls)

    #     if inspect.isclass(n_cls):
    #         res = object.__new__(n_cls)
    #     elif callable(n_cls):
    #         res = n_cls(path, *args, schema=file_format, **kwargs)
    #     else:
    #         raise RuntimeError(f"Illegal schema! {file_format} {n_cls} {path}")

    #     return res

    def __init__(self,  data=None, *args, metadata=None, working_dir=None, envs=None,  ** kwargs):
        super().__init__(data, *args, metadata=metadata, envs=envs, working_dir=working_dir, ** kwargs)

        if working_dir is None:
            working_dir = pathlib.Path.cwd()
        else:
            working_dir = pathlib.Path(working_dir)

        if isinstance(data, (pathlib.PosixPath, str)):
            self._path = working_dir/data
            data = None
        else:
            schema = self.metadata["$schema"]
            path = schema.path or ""
            self._path = working_dir/path

        if not self._path:
            raise ValueError(f"File path is not defined!")

        if self.is_writable:
            self.update(data or self.metadata.default)

    @property
    def __repr__(self):
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
