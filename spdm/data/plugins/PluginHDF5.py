from ..Collection import FileCollection
from ..Document import Document
from ..Handler import (Holder, Handler)
import h5py
import numpy
import collections
import pathlib
from typing import (Dict, Any)
from spdm.util.logger import logger

SPDM_LIGHTDATA_MAX_LENGTH = 128


class HDF5Holder(Holder):
    def __init__(self, obj,  *args, **kwargs):
        if not isinstance(obj, h5py.Group):
            obj = h5py.File(obj, *args, **kwargs)
        super().__init__(obj)


class HDF5Handler(Handler):

    def require_group(self, grp, path):
        for p in path:
            if isinstance(p, str):
                pass
            elif isinstance(p, int):
                if p < 0:
                    num = len(grp)
                    p = p % num
                p = f"__index__{p}"

            grp = grp.require_group(p)

        return grp

    def put(self, holder: HDF5Holder, path, value, *args, **kwargs):
        grp = holder.data

        path, query, fragment = self.request(path, *args, **kwargs)

        if grp is None:
            raise RuntimeError("None group")

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

    def get(self, holder: HDF5Holder, path=[], projection=None, *args, **kwargs):
        obj = holder.data

        path, query, fragment = self.request(path, *args, **kwargs)

        if obj is None:
            raise RuntimeError("None group")

        if isinstance(path, str):
            path = path.split(Handler.DELIMITER)
        elif not isinstance(path, collections.abc.Sequence):
            raise TypeError(f"Illegal path type {type(path)}! {path}")

        prefix = []

        for p in path:
            if isinstance(p, str):
                pass
            elif isinstance(p, int):
                if p < 0:
                    num = len(grp)
                    p = p % num
                p = f"__index__{p}"

            prefix.append(p)

            if p in obj:
                obj = obj[p]
            elif p in obj.attrs:
                obj = obj.attrs[p]
            else:
                raise KeyError(f"Can not find element at {'/'.join(prefix)} !")

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


def connect_hdf5(uri, *args, **kwargs):

    path = pathlib.Path(getattr(uri, "path", uri))

    return FileCollection(path, *args,
                          file_extension=".h5",
                          file_factory=HDF5Holder,
                          handler=HDF5Handler(),
                          **kwargs)


__SP_EXPORT__ = connect_hdf5
