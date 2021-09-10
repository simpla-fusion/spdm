# from spdm.data.Collection import FileCollection
import collections
import pathlib
from typing import Any, Dict

import h5py
import numpy
from spdm.data.Entry import Entry
from spdm.data.File import FileHandler, File
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger

SPDM_LIGHTDATA_MAX_LENGTH = 64


def h5_require_group(grp, path):
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


def h5_put_value(grp, path, value):
    res = None
    if path is None:
        path = []
    elif type(path) is not list:
        path = [path]

    if isinstance(value, collections.abc.Mapping):
        grp = h5_require_group(grp, path)
        for k, v in value.items():
            h5_put_value(grp, [k], v)
    elif len(path) == 0:
        raise KeyError(f"Empty path!")
    else:
        grp = h5_require_group(grp, path[:-1])
        path = path[-1]

        if isinstance(path, int):
            path = f"__index__{path}"
        # elif not isinstance(path, str):
        #     raise KeyError(path)
        if path != '' and path in grp.keys():
            del grp[path]

        if type(value) is list:
            array_value = numpy.array(value)

            if array_value.dtype.type is numpy.object_:
                grp = h5_require_group(grp, path)

                grp.attrs["__is_list__"] = True

                for idx, v in enumerate(value):
                    h5_put_value(grp, idx, v)

            elif array_value.dtype.type is numpy.unicode_:
                # h5py does not support unicode string.
                array_value = array_value.astype(h5py.special_dtype(vlen=str))
                h5_put_value(grp, path, array_value)
            else:
                h5_put_value(grp, path, array_value)
        elif type(value) is numpy.ndarray and len(value) > SPDM_LIGHTDATA_MAX_LENGTH:
            grp[path] = value
        else:  # type(value) in [str, int, float]:
            grp.attrs[path] = value

    return res


def h5_get_value(obj, path=None, projection=None):
    if obj is None:
        raise RuntimeError("None group")

    prefix = []
    if path is not None:
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
                res = [h5_get_value(obj[k]) for k in obj]
            else:
                res = {**(h5_get_value(obj.attrs)), **
                       {k: h5_get_value(obj[k]) for k in obj}}
        elif isinstance(obj, h5py.AttributeManager):
            res = {k: h5_get_value(obj[k])
                   for k in obj if not k.startswith("__")}
        elif isinstance(obj, h5py.Dataset):
            res = obj[:]
        else:
            res = obj
    elif isinstance(projection, str):
        if isinstance(obj, h5py.Group):
            res = h5_get_value(obj.attrs, projection) or h5_get_value(
                obj.get(projection, None))
        elif isinstance(obj, h5py.AttributeManager):
            res = h5_get_value(obj.get(projection, None))

    elif isinstance(obj, h5py.Group):
        if obj.attrs.get("__is_list__", False):
            res = []
        else:
            res = {**h5_get_value(obj.attrs, projection),
                   **{k: h5_get_value(obj[k]) for k, v in projection.items() if v > 0 and k in obj}}
    elif isinstance(obj, h5py.AttributeManager):
        res = {k: h5_get_value(obj[k])
               for k, v in projection.items() if v > 0 and k in obj}
    elif isinstance(obj, h5py.Dataset):
        res = obj[:]
    else:
        res = obj

    return res


def h5_dump(grp):
    return h5_get_value(grp, [])


class H5Entry(Entry):

    def __init__(self, holder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.holder = holder

    def copy(self, other):
        if isinstance(other, LazyProxy):
            other = other.__real_value__()
        elif isinstance(other, Entry):
            other = other.entry.__real_value__()
        self.put(None, other)

    def put(self, path, value, *args, **kwargs):
        return h5_put_value(self.holder, path, value)

    def get(self, path=[], projection=None, *args, **kwargs):
        return h5_get_value(self.holder, path, projection=projection)

    def dump(self):
        return h5_dump(self.holder)

    def iter(self,  path, *args, **kwargs):
        raise NotImplementedError()


class H5File(FileHandler):
    def __init__(self,  *args,  **kwargs):
        super().__init__(*args,   **kwargs)
        path = self._metadata.get("path", "")
        mode = self._metadata.get("mode", "r")
        mode = ''.join([(m.name) for m in list(File.Mode) if m & mode])
        try:
            fid = h5py.File(path,  mode=mode)
        except OSError as error:
            raise FileExistsError(f"Can not open file {path}! {error}")
        else:
            logger.debug(f"Open HDF5 File {path} mode={mode}")

        self._entry = H5Entry(fid)

    def read(self, lazy=True) -> Entry:
        return self._entry

    def write(self, *args, **kwargs):
        self._entry.put([], *args, **kwargs)


# class HDF5Collection(FileCollection):
#     def __init__(self, uri, *args, **kwargs):
#         super().__init__(uri, *args,
#                          file_extension=".h5",
#                          file_factory=lambda *a, **k: H5File(*a, **k),
#                          ** kwargs)

__SP_EXPORT__ = H5File
