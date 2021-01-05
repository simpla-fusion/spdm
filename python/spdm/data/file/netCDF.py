import collections
import logging
import os
import pathlib

import netCDF4 as nc
import numpy as np
from spdm.util.logger import logger
from spdm.util.utilities import whoami

from ..File import File
from ..Node import Node


class NetCDFNode(Node):
    def __init__(self, data, *args,  **kwargs):
        super().__init__(data, *args, **kwargs)

    def get(self,  path, *args, **kwargs):
        raise NotImplementedError(path)

    def get_value(self, path, *args, **kwargs):
        if isinstance(path, str):
            path = path.split('.')

        path = self.prefix+path

        obj = self._holder

        for idx in path:
            if obj is None:
                break
            elif isinstance(obj, nc.Dataset):
                obj = obj.groups.get(idx, None) or\
                    obj.variables.get(idx, None)
            elif isinstance(obj, nc.Group):
                obj = obj[idx]
            elif isinstance(obj, nc.Variable) and (type(idx) in [int, slice, tuple]):
                obj = obj[idx]
            elif hasattr(obj, idx):
                obj = getattr(obj, idx)
            else:
                raise IndexError(idx)
        if isinstance(obj, nc.Variable):
            obj = obj[:]
        return obj

    def write(self, path, d):
        raise NotImplementedError

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

    def put(self, path, value, *args, **kwargs):
        if isinstance(path, str):
            path = pathlib.Path(path)
        return self._update(self._holder, path, value, *args, **kwargs)

    def _fetch(self, obj, path=None):
        if path is not None:
            if not isinstance(path, str):
                path = str(path).strip("/").replace("/", ".")
            try:
                return obj.getncattr(path)
            except AttributeError:
                return self._fetch(obj.get_variables_by_attributes(name=path)[0])
        elif type(obj) in [str, int, float]:
            return obj
        elif isinstance(obj, nc._nc.Variable):
            return obj[:]
        else:
            raise NotImplementedError(whoami(self))

    def do_fetch(self, path):
        if isinstance(path, str):
            path = pathlib.Path(path)
        return self._fetch(self._open(), path)

    def iter(self,  path, *args, **kwargs):
        raise NotImplementedError
        # for spath in PathTraverser(path):
        #     for child in self.xpath(spath).evaluate(self._holder):
        #         if child.tag is not _XMLComment:
        #             yield self._convert(child, path=spath)


class FileNetCDF(File):
    def __init__(self, data=None, *args, mode="r", **kwargs):
        super().__init__(data, *args,   **kwargs)
        self._root = None

    @property
    def root(self):
        if self._root is None:
            self._root = nc.Dataset(self.path, self.mode)
        return NetCDFNode(self._root)


__SP_EXPORT__ = FileNetCDF
