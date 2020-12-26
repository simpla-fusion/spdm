import collections
import logging
import os
import pathlib
import re
import uuid
from functools import reduce
from typing import Any, Dict


import numpy as np

import netCDF4 as netCDF4
from spdm.util.logger import logger
from spdm.util.utilities import whoami
from spdm.util.LazyProxy import LazyProxy
from ..File import File

__plugin_spec__ = {
    "name": "netcdf",
    "filename_pattern": ["*.nc"],
    "filename_extension": "nc"
}


class FileNetCDF(File):
    def __init__(self, desc, value=None, *args, mode="r", **kwargs):
        super().__init__(desc, value, *args,   **kwargs)
        # if isinstance(fp, str):
        #     self._file_path = pathlib.Path(fp)
        # elif isinstance(fp, os.PathLike):
        #     self._file_path = fp
        # else:
        #     raise TypeError(
        #         f"expected str, bytes or os.PathLike, not {type(fp)}")

        # self._grp = netCDF4.Dataset(self._file_path, mode)
        # self._mode = mode

    def read(self, path):
        if isinstance(path, str):
            path = [path]

        obj = self._grp

        for idx in path:
            if obj is None:
                break
            elif isinstance(obj, netCDF4.Dataset):
                obj = obj.groups.get(idx, None) or\
                    obj.variables.get(idx, None)
            elif isinstance(obj, netCDF4.Group):
                obj = obj[idx]
            elif isinstance(obj, netCDF4.Variable) and (type(idx) in [int, slice, tuple]):
                obj = obj[idx]
            elif hasattr(obj, idx):
                obj = getattr(obj, idx)
            else:
                raise IndexError(idx)
        if isinstance(obj, netCDF4.Variable):
            obj = obj[:]
        return obj

    def write(self, path, d):
        raise NotImplementedError

    def _open(self):
        if self._grp is None:
            self._grp = nc.Dataset(self._file_path, self._mode)
        return self._grp

    def _update(self, grp, path, value, **kwargs):
        if grp is None:
            raise RuntimeError("None group")

        if path is None:
            raise RuntimeError(f"None path")

        if isinstance(value, collections.abc.Mapping):
            for k, v in value.items():
                self._update(grp, path/k, v)
            return

        prefix = str(path.parent).replace("/", ".")

        attr_name = path.name
        if len(prefix) > 0 and prefix != '.':
            grp = grp.createGroup(prefix)
        if type(value) in [str, int, float]:
            setattr(grp, attr_name, value)
        elif not isinstance(value, np.ndarray):
            raise f"Unsupported data type {type(value)}"
        else:
            logger.debug(f"insert ndarray {value.shape} {value.dtype.str}")
            dimensions = list()
            for i in range(len(value.shape)):
                d_name = f"_{attr_name}_d{i}"
                grp.createDimension(d_name, value.shape[i])
                dimensions.append(d_name)
            data = grp.createVariable(attr_name, value.dtype.str, dimensions)
            data[:] = value

        return
        # if prefix.is_empty():

        #     return self._update(grp.require_group(path[:pos]),
        #                         path[pos + 1:], value)

        # if path in grp.keys() and path != '/':
        #     del grp[path]

        # if type(value) in [str, int, float]:
        #     grp.attrs[path] = value
        # elif type(value) is list:
        #     value = numpy.array(value)
        #     # NOTE(salmon 2019.7.5):h5py does not support unicode string.
        #     if value.dtype.type is numpy.unicode_:
        #         value = value.astype(h5py.special_dtype(vlen=str))

        #     if len(value) < spdm_LIGHTDATA_MAX_LENGTH:
        #         grp.attrs[path] = value
        #     else:
        #         grp[path] = value

        # elif type(value) is dict:
        #     g = grp.require_group(path)
        #     for k, v in value.items():
        #         self._update(g, k, v)
        # elif type(value) is numpy.ndarray:
        #     grp[path] = value
        # else:
        #     raise RuntimeError(f"Unsupported data type {type(value)}")

    def do_update(self, path, value):
        if isinstance(path, str):
            path = pathlib.Path(path)
        return self._update(self._open(), path, value)

    def _fetch(self, obj, path=None):
        if path is not None:
            if not isinstance(path, str):
                path = str(path).strip("/").replace("/", ".")
            try:
                logger.debug(path)
                return obj.getncattr(path)
            except AttributeError:
                return self._fetch(
                    obj.get_variables_by_attributes(name=path)[0])
        elif type(obj) in [str, int, float]:
            return obj
        elif isinstance(obj, nc._netCDF4.Variable):
            return obj[:]
        else:
            raise NotImplementedError(whoami(self))

    def do_fetch(self, path):
        if isinstance(path, str):
            path = pathlib.Path(path)
        return self._fetch(self._open(), path)

    # def fetch_if(self,
    #              projection: Dict[str, Any],
    #              predicate: Dict[str, Any] = None):
    #     raise NotImplementedError(whoami(self))

    # def update_if(self,
    #               update: Dict[str, Any],
    #               predicate: Dict[str, Any] = None):
    #     raise NotImplementedError(whoami(self))

    def do_delete(self, path, *args, **kwargs):
        raise NotImplementedError(whoami(self))

    def exists(self, path, *args, **kwargs):
        raise NotImplementedError(whoami(self))

    def dir(self, *args, **kwargs):

        raise NotImplementedError(whoami(self))


_plugin_spec__ = {
    "name": "netcdf",
    "filename_pattern": ["*.nc"],
    "filename_extension": "nc",
    "support_data_type": [int, float, str, dict]
}


# def connect(*args, **kwargs):
#     return NetCDFEntry(*args, **kwargs)


# def read(fid, *args, **kwargs) -> Dict[str, Any]:
#     conn = NetCDFEntry(fid, *args, **kwargs)
#     return ReadablePointer(conn.read)


# def write(fid, data: Dict[str, Any], *args, **kwargs):
#     return NetCDFEntry(fid,  *args, template=None, **kwargs).write(data)

__SP_EXPORT__ = FileNetCDF
