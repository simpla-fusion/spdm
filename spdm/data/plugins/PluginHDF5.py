from ..Collection import Collection
from ..Entry import Entry
from ..Document import Document
import pathlib
import h5py
import numpy
import collections
from typing import (Dict, Any)
from spdm.util.logger import logger
from spdm.util.LazyProxy import LazyProxy


class EntryHDF5Handler(LazyProxy.Handler):

    def require_group(self, grp, path):
        for p in path:
            if isinstance(p, str):
                pass
            elif p < 0:
                p = f"_id_{len(grp)}"
            else:
                p = f"_id_{p}"
            logger.debug(p)
            grp = grp.require_group(p)

        return grp

    def put(self, grp, path, value, **kwargs):
        if grp is None:
            raise RuntimeError("None group")

        if isinstance(path, str):
            path = path.split(LazyProxy.DELIMITER)
        elif not isinstance(path, collections.abc.Sequence):
            raise TypeError(f"Illegal path type {type(path)}! {path}")

        if isinstance(value, collections.abc.Mapping):
            grp = self.require_group(grp, path)
            for k, v in value.items():
                # TODO: handler operators 'k'
                self.put(grp, k, v)

        elif type(value) is list:
            array_value = numpy.array(value)

            if array_value.dtype.type is numpy.object_:
                grp = self.require_group(grp, path)

                for idx,v in enumerate(value):
                    self.put(grp, [idx], v)

            elif array_value.dtype.type is numpy.unicode_:
                # h5py does not support unicode string.
                array_value = array_value.astype(h5py.special_dtype(vlen=str))
                self.put(grp, path, array_value)
            else:
                self.put(grp, path, array_value)

        elif len(path) > 0:
            grp = self.require_group(grp, path[:-1])
            path = path[-1]

            if path != '' and path in grp.keys():
                del grp[path]

            if type(value) is numpy.ndarray and len(value) > SPDM_LIGHTDATA_MAX_LENGTH:
                grp[path] = value
            else:  # type(value) in [str, int, float]:
                grp.attrs[path] = value
        else:
            raise TypeError(f"Can not put {type(value)} to group!")

            # else:
            #     raise RuntimeError(f"Unsupported data type {type(value)}")

    def get(self, obj, projection=None):
        if projection is None:
            if isinstance(obj, h5py.Group):
                return {**self.get(obj.attrs),
                        **{k: self.get(obj[k]) for k in obj}}
            elif isinstance(obj, h5py.AttributeManager):
                return {k: self.get(obj[k]) for k in obj}
            else:
                return obj
        elif isinstance(projection, str):
            if isinstance(obj, h5py.Group):
                return self.get(obj.attrs, projection) or self.get(obj.get(projection, None))
            elif isinstance(obj, h5py.AttributeManager):
                return self.get(obj.get(projection, None))
        elif isinstance(obj, h5py.Group):
            return {**self.get(obj.attrs, projection),
                    **{k: self.get(obj[k])
                       for k, v in projection.items() if v > 0 and k in obj}}
        elif isinstance(obj, h5py.AttributeManager):
            return {k: self.get(obj[k])
                    for k, v in projection.items() if v > 0 and k in obj}
        else:
            return obj


class DocumentHDF5(Document):
    def __init__(self, path, *args, mode="w", **kwargs):
        super().__init__(None, *args, **kwargs)
        self._mode = mode
        self._file = h5py.File(path, mode=mode)
        self._handler = EntryHDF5Handler()
        logger.debug(f"Open HDF5 File: {path} mode=\"{mode}\"")

    @property
    def entry(self):
        return LazyProxy(self._file, handler=self._handler)

    def update(self, d: Dict[str, Any]):
        return self._handler.put(self._file, [], d)

    def fetch(self, proj: Dict[str, Any] = None):
        return self._handler.get(self._file, proj)


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

        logger.debug(f"Open HDF5 Collection : {self._path}")

    # mode in ["", auto_inc  , glob ]
    def get_filename(self, d, mode=""):
        if self._filename_pattern is not None:
            fname = self._filename_pattern(self._path, d, mode)
        elif mode == "auto_inc":
            fnum = len(list(self._path.parent.glob(self._path.name.format(_id="*"))))
            fname = (self._path.name or "{_id}.h5").format(_id=fnum)
        elif mode == "glob":
            fname = (self._path.name or "{_id}.h5").format(_id="*")
        else:
            try:
                fname = (self._path.name or "{_id}.h5").format_map(d)
            except KeyError:
                fname = None

        return fname

    def insert_one(self, data=None,  **kwargs):
        doc = DocumentHDF5(self._path.with_name(self.get_filename(data or kwargs, mode="auto_inc")), mode="w")
        doc.update(data or kwargs)
        return doc

    def find_one(self, predicate=None, projection=None, **kwargs):
        fname = self.get_filename(predicate or kwargs)
        doc = None
        if fname is not None:
            doc = DocumentHDF5(self._path.with_name(fname), mode="r")
        else:
            for fp in self._path.parent.glob(self.get_filename(predicate or kwargs, mode="glob")):
                if not fp.exists():
                    continue
                doc = DocumentHDF5(fp, mode="r")
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
        raise NotImplementedError()


def connect_hdf5(uri, *args, **kwargs):

    return CollectionHDF5(uri.path, *args, **kwargs)


__SP_EXPORT__ = connect_hdf5
