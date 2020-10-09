import contextlib
import io
import pathlib
import tempfile
import uuid
import shutil
from spdm.data.DataEntry import DataEntry
from spdm.util.SpURI import SpURI, urisplit


class FileEntry(DataEntry):
    """ Default entry for file-like object
    """

    def __init__(self, path, *args,
                 mode='r',
                 buffering=-1,
                 encoding=None,
                 errors=None,
                 newline=None,
                 is_temp=False,
                 suffix=".yaml",
                 prefix=None,
                 dir=None,
                 ** kwargs):

        if isinstance(dir, str):
            dir = pathlib.Path(dir)
        if path is None:
            path = f"{prefix or ''}{uuid.uuid1().hex()}{suffix or ''}"
        if isinstance(path, SpURI):
            path = dir/path.path
        if isinstance(path, str):
            if dir is None:
                path = pathlib.Path(path)
            else:
                path = dir/path

        if not isinstance(path, pathlib.Path):
            raise TypeError(
                f"Entry is not file-like or string or Path! {type(path)}")

        # if is_temp or (entry is None):
        #     entry = tempfile.TemporaryFile(
        #         mode=mode,
        #         buffering=buffering,
        #         encoding=encoding,
        #         newline=newline,
        #         suffix=suffix,
        #         prefix=prefix,
        #         dir=dir)

        #     entry = entry.open(
        #         mode=mode,
        #         buffering=buffering,
        #         encoding=encoding,
        #         errors=errors,
        #         newline=newline)
        # elif isinstance(entry, io.IOBase):
        #     pass
        # else:

        super().__init__(path, *args, ** kwargs)

        self._mode = mode
        self._buffering = buffering
        self._encoding = encoding
        self._errors = errors
        self._newline = newline

    def flush(self, *args, **kwargs):
        pass

    def copy(self, path=None):
        if path is None:
            path = f"{self._uri.path.stem}_copy{self._uri.path.suffix}"
        elif isinstance(path, str):
            path = pathlib.Path(path)
            if path.is_dir() and path != self._uri.path.parent:
                path = path/self._uri.path.name
        shutil.copy(self._uri, path.as_posix())
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
        if isinstance(self._uri, pathlib.Path):
            path = self._uri
        else:
            o = urisplit(self._uri)
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
        old_d = self.read()
        old_d.update(d)
        self.write(old_d)


__SP_EXPORT__ = FileEntry
