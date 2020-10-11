from ..Collection import FileCollection
from ..Document import Document
from ..Handler import Handler
import h5py
import numpy
import collections
import pathlib
from typing import (Dict, Any)
from spdm.util.logger import logger

SPDM_LIGHTDATA_MAX_LENGTH = 128


class HDF5Handler(Handler):

    def require_group(self, grp, path):
        for p in path:
            if isinstance(p, str):
                pass
            elif isinstance(p, int):
                if p < 0:
                    num = len(grp)
                    p = p % num
                p = f"__id__{p}"

            grp = grp.require_group(p)

        return grp

    def put(self, grp, path, value, **kwargs):
        if grp is None:
            raise RuntimeError("None group")

        if isinstance(path, str):
            path = path.split(Handler.DELIMITER)
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

                grp.attrs["__is_list__"] = True

                for idx, v in enumerate(value):
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

    def get(self, obj, path=[], projection=None):
        if obj is None:
            raise RuntimeError("None group")

        if isinstance(path, str):
            path = path.split(Handler.DELIMITER)
        elif not isinstance(path, collections.abc.Sequence):
            raise TypeError(f"Illegal path type {type(path)}! {path}")

        for p in path:
            if isinstance(p, str):
                pass
            elif isinstance(p, int):
                if p < 0:
                    num = len(grp)
                    p = p % num
                p = f"__id__{p}"

            if p in obj:
                obj = obj[p]
            elif p in obj.attrs:
                obj = obj.attrs[p]

        if projection is None:
            if isinstance(obj, h5py.Group):
                if obj.attrs.get("__is_list__", False):
                    res = [self.get(obj[k]) for k in obj]
                else:
                    res = {**self.get(obj.attrs), **{k: self.get(obj[k]) for k in obj}}
            elif isinstance(obj, h5py.AttributeManager):
                res = {k: self.get(obj[k]) for k in obj if not k.startswith("__")}
            else:
                res = obj
        elif isinstance(projection, str):
            if isinstance(obj, h5py.Group):
                res = self.get(obj.attrs, projection) or self.get(obj.get(projection, None))
            elif isinstance(obj, h5py.AttributeManager):
                res = self.get(obj.get(projection, None))

        elif isinstance(obj, h5py.Group):
            logger.debug(obj.attrs)
            if obj.attrs.get("__is_list__", False):
                res = []
            else:
                res = {**self.get(obj.attrs, projection),
                       **{k: self.get(obj[k]) for k, v in projection.items() if v > 0 and k in obj}}
        elif isinstance(obj, h5py.AttributeManager):
            res = {k: self.get(obj[k]) for k, v in projection.items() if v > 0 and k in obj}
        else:
            res = obj

        return res


def connect_hdf5(uri, *args, filename_pattern="{_id}.h5", handler=None, **kwargs):

    path = pathlib.Path(getattr(uri, "path", uri))

    return FileCollection(path, *args,
                          filename_pattern=filename_pattern,
                          document_factory=lambda fpath, mode: Document(
                              root=h5py.File(fpath, mode=mode),
                              handler=handler or HDF5Handler()
                          ),
                          **kwargs)


__SP_EXPORT__ = connect_hdf5
