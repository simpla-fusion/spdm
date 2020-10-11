from ..Collection import Collection
from ..Entry import Entry
from ..Document import Document
import pathlib
import h5py
import numpy
import collections
from typing import (Dict, Any)
from spdm.util.logger import logger


class DocumentHDF5(Document):
    def __init__(self, path, *args, mode="w", **kwargs):
        super().__init__(None, *args, **kwargs)
        self._mode = mode
        logger.debug(path)
        self._file = h5py.File(path, mode=mode)
        self._grp_root = self._file

    def _open(self):
        if self._grp_root is None:
            self._grp_root = h5py.File(self._file_path)
        return self._grp_root

    def _insert(self, grp, path, value, **kwargs):
        if grp is None:
            raise RuntimeError("None group")

        if path is None:
            raise RuntimeError(f"None path")
        pos = path.rfind('/')
        if pos > 0:
            return self._insert(grp.require_group(path[:pos]), path[pos + 1:], value)

        if path in grp.keys() and path != '/':
            del grp[path]

        if type(value) in [str, int, float]:
            grp.attrs[path] = value
        elif type(value) is list:
            value = numpy.array(value)
            # h5py does not support unicode string.
            if value.dtype.type is numpy.unicode_:
                value = value.astype(h5py.special_dtype(vlen=str))

            if len(value) < SPDM_LIGHTDATA_MAX_LENGTH:
                grp.attrs[path] = value
            else:
                grp[path] = value

        elif type(value) is dict:
            g = grp.require_group(path)
            for k, v in value.items():
                self._insert(g, k, v)
        elif type(value) is numpy.ndarray:
            grp[path] = value
        else:
            raise RuntimeError(f"Unsupported data type {type(value)}")
        return

    def _fetch(self, obj, projection=None):
        if projection is None:
            if isinstance(obj, h5py.Group):
                return {**self._fetch(obj.attrs),
                        **{k: self._fetch(obj[k]) for k in obj}}
            elif isinstance(obj, h5py.AttributeManager):
                return {k: self._fetch(obj[k]) for k in obj}
            else:
                return obj
        elif isinstance(projection, str):
            if isinstance(obj, h5py.Group):
                return self._fetch(obj.attrs, projection) \
                    or self._fetch(obj.get(projection, None))
            elif isinstance(obj, h5py.AttributeManager):
                return self._fetch(obj.get(projection, None))
        elif isinstance(obj, h5py.Group):
            return {**self._fetch(obj.attrs, projection),
                    **{k: self._fetch(obj[k])
                       for k, v in projection.items() if v > 0 and k in obj}}
        elif isinstance(obj, h5py.AttributeManager):
            return {k: self._fetch(obj[k])
                    for k, v in projection.items() if v > 0 and k in obj}
        else:
            return obj

    def update(self, d: Dict[str, Any]):
        raise NotImplementedError()
        # return self._insert(self._open(), "/", d)

    def fetch(self, proj: Dict[str, Any] = None):
        return self._fetch(self._open(), proj)

    def delete(self, path, *args, **kwargs):
        pass

    def exists(self, path, *args, **kwargs):
        raise NotImplementedError(whoami(self))

    def dir(self, *args, **kwargs):
        self.pull_cache()
        raise NotImplementedError(whoami(self))


class CollectionHDF5(Collection):

    def __init__(self, path, *args, filename_pattern=None, mode="rw",  **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = mode

        self._path = pathlib.Path(path).resolve().expanduser()

        self._filename_pattern = filename_pattern

        if not self._path.parent.exists():
            if "w" not in mode:
                raise RuntimeError(f"Can not make dir {self._path}")
            else:
                self._path.parent.mkdir()
        elif not self._path.parent.is_dir():
            raise NotADirectoryError(self._path.parent)

    def get_filename(self, d, auto_inc=False):
        if self._filename_pattern is not None:
            fname = self._filename_pattern(self._path, d, auto_inc)
        else:
            try:
                fname = (self._path.name or "{_id}.h5").format_map(d)
            except KeyError:
                if auto_inc:
                    fname = (self._path.name or "{_id}.h5").format(
                        _id=len(list(self._path.parent.glob(self._path.name.format(_id="*")))))
        if fname == "":
            raise FileNotFoundError(d)
        return self._path.parent.with_name(fname)

    def insert_one(self, data=None,  **kwargs):
        doc = DocumentHDF5(self.get_filename(data or kwargs, auto_inc=True), mode="w")
        # doc.update(data)
        return doc

    def find_one(self, predicate=None, projection=None, **kwargs):

        doc = DocumentHDF5(self.get_filename(predicate or kwargs), mode="r")

        if projection is not None:
            raise NotImplementedError()

        return doc

    def update_one(self, predicate, update,  *args, **kwargs):
        raise NotImplementedError()

    def delete_one(self, predicate,  *args, **kwargs):
        raise NotImplementedError()

    def count(self, predicate=None,   *args, **kwargs) -> int:
        raise NotImplementedError()


def connect_hdf5(uri, *args, **kwargs):

    return CollectionHDF5(uri.path, *args, **kwargs)


__SP_EXPORT__ = connect_hdf5
