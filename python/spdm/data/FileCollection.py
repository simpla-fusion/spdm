import collections
import inspect
import pathlib
import re
import urllib
from typing import Any, Dict, List, NewType, Tuple

from ..util.logger import logger
from ..util.uri_utils import uri_merge, uri_split
from .Collection import Collection
from .Directory import Directory
from .Document import Document
from .File import File
from .Entry import Entry


class FileCollection(Directory, Collection):

    def __init__(self, uri, *args,
                 file_extension=".dat",
                 file_factory=None,
                 **kwargs):

        super().__init__(uri, *args, **kwargs)

        if isinstance(uri, str):
            uri = uri_split(uri)

        path = getattr(uri, "path", ".").replace("*", "{_id}")

        if isinstance(path, str) and path.find("{_id}") < 0:  # not path.endswith(file_extension):
            path = f"{path}{{_id}}{file_extension}"

        self._path = pathlib.Path(path).resolve().expanduser()

        logger.debug(self._path)

        if self._path.suffix == '':
            self._path = self._path.with_suffix(file_extension)

        if "{_id}" not in self._path.stem:
            self._path = self._path.with_name(f"{self._path.stem}{{_id}}{self._path.suffix}")

        if not self._path.parent.exists():
            if "w" not in self._mode:
                raise RuntimeError(f"Can not make dir {self._path}")
            else:
                self._path.parent.mkdir()
        elif not self._path.parent.is_dir():
            raise NotADirectoryError(self._path.parent)

        self._file_factory = file_factory or File

    def guess_id(self, d, auto_inc=True):
        fid = super().guess_id(d, auto_inc=auto_inc)

        if fid is None and auto_inc:
            fid = self.count()

        return fid

    def guess_filepath(self, *args, **kwargs):
        return self._path.with_name(self._path.name.format(_id=self.guess_id(*args, **kwargs)))

    def open_document(self, fid, mode=None):
        fpath = self.guess_filepath({"_id": fid})
        logger.debug(f"Open Document: {fpath} mode=\"{ mode or self.metadata.mode}\"")
        return Document(root=self._file_factory(fpath, mode or self.metadata.mode), fid=fid, envs=self.envs, handler=self._handler)

    def insert_one(self, data=None, *args,  **kwargs):
        doc = self.open_document(self.guess_id(data or kwargs, auto_inc=True), mode="w")
        doc.update(data or kwargs)
        return doc

    def find_one(self, predicate=None, projection=None, **kwargs) -> Entry:
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

__SP_EXPORT__ = FileCollection
