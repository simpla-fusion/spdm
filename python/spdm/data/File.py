from __future__ import annotations

import collections
import collections.abc
import pathlib
import typing

from ..utils.logger import logger
from ..utils.tags import _undefined_
from ..utils.uri_utils import URITuple, uri_split
from .Collection import Collection, InsertOneResult
from .Connection import Connection
from .Entry import Entry


class File(Connection):
    """
        File like object
    """

    def __init__(self, url: str | pathlib.Path | URITuple, *args, format=None, default_format=None, **kwargs):
        if self.__class__ is not File:
            pass
        elif not isinstance(url, (str, URITuple, pathlib.Path)):
            raise ValueError(f"Unknown URL: {url}")
        else:

            if format is not None:
                pass
            elif isinstance(url, dict):
                format = url.get("$class", None)
            elif isinstance(url,   pathlib.PosixPath):
                format = url.suffix[1:].upper()
            elif isinstance(url, (str, URITuple)):
                uri = uri_split(url)
                if isinstance(uri.protocol, str) and uri.protocol != "file":
                    format = uri.protocol
                else:
                    format = pathlib.PosixPath(uri.path).suffix[1:].upper()

            if format == ".":
                format = "text"

            plugin_list = []

            if format is not None:
                plugin_list.append(format)

            if default_format is not None:
                plugin_list.append(default_format)

            plugin_list = [(f"spdm.plugins.data.plugin_{s.lower()}#FILEPLUGIN{s.lower()}" if isinstance(s, str) and "." not in s else s)
                           for s in plugin_list]

            super().__dispatch__init__(plugin_list, self, url, *args, **kwargs)

            return

        super().__init__(url, *args, **kwargs)

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


@Collection.register(["localdb", "FileCollection"])
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
