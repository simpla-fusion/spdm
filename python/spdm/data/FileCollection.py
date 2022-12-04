import collections
import inspect
import pathlib
import re
import urllib
from typing import Any, Dict, List, NewType, Tuple, Union

from ..util.logger import logger
from ..util.uri_utils import uri_merge, uri_split
from .Collection import Collection
from .Directory import Directory
from .Document import Document
from .File import File
from .Entry import Entry
from ..common.tags import _undefined_
from .Collection import InsertOneResult


class FileCollection(Directory):

    def __init__(self, *args, file_name=_undefined_, ** kwargs):
        """
        Example:
            file_name="{*}"
        """

        super().__init__(*args, **kwargs)

        if file_name not in ("", _undefined_, None):
            self._file_name = file_name
        if self.glob == "":
            self._file_name = "{_id_}"
        else:
            self._file_name = self.glob.replace("*", "{_id_}")

        # if self._path.suffix == '':
        #     self._path = self._path.with_suffix(extension)

        # if "{_id}" not in self._path.stem:
        #     self._path = self._path.with_name(f"{self._path.stem}{{_id}}{self._path.suffix}")

        # if not self._path.parent.exists():
        #     if "w" not in self._mode:
        #         raise RuntimeError(f"Can not make dir {self._path}")
        #     else:
        #         self._path.parent.mkdir()
        # elif not self._path.parent.is_dir():
        #     raise NotADirectoryError(self._path.parent)

    def guess_id(self, d, auto_inc=True):
        fid = super().guess_id(d, auto_inc=auto_inc)

        if fid is None and auto_inc:
            fid = self.count()

        return fid

    def guess_filepath(self, *args, **kwargs):
        return self.path/self._file_name.name.format(_id=self.guess_id(*args, **kwargs))

    def open_document(self, fid, mode=None):
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


__SP_EXPORT__ = FileCollection
