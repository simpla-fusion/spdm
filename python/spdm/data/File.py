from __future__ import annotations

import collections
import collections.abc
import pathlib
import typing

from ..utils.logger import logger
from ..utils.sp_export import sp_load_module
from ..utils.tags import _undefined_
from ..utils.uri_utils import URITuple, uri_split
from .Collection import Collection, InsertOneResult
from .Connection import Connection
from .Entry import Entry


class File(Connection):
    """
        File like object
    """
    @classmethod
    def __dispatch__init__(cls, name_list, self, path, *args, **kwargs) -> None:
        if name_list is None:
            n_cls_name = ''
            if "format" in kwargs:
                n_cls_name = kwargs.get("format")
            elif isinstance(path, collections.abc.Mapping):
                n_cls_name = path.get("$class", None)
            elif isinstance(path,   pathlib.PosixPath):
                n_cls_name = path.suffix[1:].upper()
            elif isinstance(path, (str, URITuple)):
                uri = uri_split(path)
                if isinstance(uri.format, str):
                    n_cls_name = uri.format
                else:
                    n_cls_name = pathlib.PosixPath(uri.path).suffix[1:].upper()
            if n_cls_name == ".":
                n_cls_name = ".text"

            #  f"{cls._plugin_prefix}{n_cls_name}#{n_cls_name}{cls.__name__}"

            name_list = [f"spdm.plugins.data.Plugin{n_cls_name}#{n_cls_name}File"]

        if name_list is None or len(name_list) == 0:
            return super().__init__(self, path, *args, **kwargs)
        else:
            return super().__dispatch__init__(name_list, self, path, *args, **kwargs)

    def __init__(self, path: str | pathlib.Path, *args,  **kwargs):
        if self.__class__ is File:
            File.__dispatch__init__(None, self, path, *args, **kwargs)
            return
        super().__init__(path, *args, **kwargs)

    @property
    def mode_str(self) -> str:
        return File.MOD_MAP.get(self.mode, "r")

    @property
    def entry(self) -> Entry:
        if self.is_readable:
            return self.read()
        else:
            return self.write()

    def read(self, lazy=False) -> Entry:
        if self._holder is None:
            self.open()
        return self._holder.read(lazy=lazy)

    def write(self, *args, **kwargs):
        if not self.is_open:
            self.open()
        self._holder.write(*args, **kwargs)

    def __enter__(self) -> File:
        return super().__enter__()

    def read(self, lazy=False) -> Entry:
        raise NotImplementedError()

    def write(self, data, lazy=False) -> Entry:
        raise NotImplementedError()


class FileCollection(Collection):

    def __init__(self, *args, glob: typing.Optional[str] = None, ** kwargs):
        """
        Example:
            file_name="{*}"
        """

        super().__init__(*args, **kwargs)

        if glob is not _undefined_:
            self._glob = glob
        else:
            parts = pathlib.Path(self.uri.path).parts

            idx, _ = next(filter(lambda s: '{' in s[1], enumerate(parts)))

            self.uri.path = pathlib.Path(*list(parts)[:idx])

            self._glob = "/".join(parts[idx:])

    @property
    def glob(self) -> str:
        return self._glob

    def guess_id(self, d, auto_inc=True):
        fid = super().guess_id(d, auto_inc=auto_inc)

        if fid is None and auto_inc:
            fid = self.count()

        return fid

    def guess_filepath(self, **kwargs) -> pathlib.Path:
        return self.path/self._glob.format(**kwargs)

    def open_document(self, fid, mode=None) -> Entry:
        fpath = self.guess_filepath({"_id_": fid})
        logger.debug(f"Open Document: {fpath} mode=\"{ mode or self.mode}\"")
        return File(fpath, mode=mode if mode is not _undefined_ else self.mode).entry

    def insert_one(self, predicate, *args,  **kwargs) -> InsertOneResult:
        doc = self.open_document(self.guess_id(predicate or kwargs, auto_inc=True))

        return doc

    def find_one(self, predicate, projection=None, **kwargs) -> Entry:
        fpath = self.guess_filepath(predicate or kwargs)

        doc = None
        if fpath.exists():
            doc = self.open_document(fpath)
        else:
            for fp in self._path.parent.glob(self._path.name.format(_id="*")):
                if not fp.exists():
                    continue
                doc = self.open_document(fp, mode="r")
                if doc.check(predicate):
                    break
                else:
                    doc = None

        if projection is not None:
            raise NotImplementedError()

        return doc

    def update_one(self, predicate, update,  *args, **kwargs):
        raise NotImplementedError()

    def delete_one(self, predicate,  *args, **kwargs):
        raise NotImplementedError()

    def count(self, predicate=None,   *args, **kwargs) -> int:
        if predicate is None:
            logger.warning("NOT IMPLEMENTED! count by predicate")

        return len(list(self._path.parent.glob(self._path.name.format(_id="*"))))


class CollectionLocalFile(Collection):
    """
        Collection of local files.
    """

    def __init__(self,   *args, file_format=None, mask=None,   **kwargs):
        super().__init__(*args, schema="local", **kwargs)

        logger.debug(self.metadata)

        self._path = pathlib.Path(self.metadata.get("authority", "") +
                                  self.metadata.get("path", ""))  # .replace("*", Collection.ID_TAG)

        self._file_format = file_format

        prefix = self._path

        while "*" in prefix.name or "{" in prefix.name:
            prefix = prefix.parent

        self._prefix = prefix
        self._filename = self._path.relative_to(self._prefix).as_posix()

        if "x" in self._mode:
            self._prefix.mkdir(parents=True, exist_ok=True, mode=mask or 0o777)
        elif not self._prefix.is_dir():
            raise NotADirectoryError(self._prefix)
        elif not self._prefix.exists():
            raise FileNotFoundError(self._prefix)

        self._path = self._path.as_posix().replace("*", "{id:06}")

    @property
    def next_id(self):
        pattern = self._filename if "*" in self._filename else "*"
        return len(list(self._prefix.glob(pattern)))

    def guess_path(self, *args, fid=None, **kwargs):
        return self._path.format(id=fid or self.next_id, **kwargs)

    def find_one(self, *args, projection=None, **kwargs):
        return File(self.guess_path(*args, **kwargs), mode=self.mode).fetch(projection)

    def insert_one(self, *args, projection=None, **kwargs):
        return File(self.guess_path(*args, **kwargs), mode="x")

    def update_one(self, predicate, update,  *args, **kwargs):
        raise NotImplementedError()

    def delete_one(self, predicate,  *args, **kwargs):
        raise NotImplementedError()
