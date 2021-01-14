
import collections
import logging
import os
import pathlib
import re
import uuid
from functools import reduce
from typing import Any, Dict

import h5py
import numpy

from ..File import File
from ..Node import Node

from spdm.util.logger import logger
from spdm.util.utilities import whoami


class HDF5Node(Node):

    def __init__(self, holder,  *args, mode="w", **kwargs):
        super().__init__(holder,  *args, **kwargs)

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
            return self._insert(grp.require_group(path[:pos]),
                                path[pos + 1:], value)

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
        return self._insert(self._open(), "/", d)

    def fetch(self, proj: Dict[str, Any] = None):
        return self._fetch(self._open(), proj)

    # def fetch_if(self,
    #              projection: Dict[str, Any],
    #              predicate: Dict[str, Any] = None):
    #     raise NotImplementedError(whoami(self))

    # def update_if(self,
    #               update: Dict[str, Any],
    #               predicate: Dict[str, Any] = None):
    #     raise NotImplementedError(whoami(self))

    def delete(self, path, *args, **kwargs):
        pass

    def exists(self, path, *args, **kwargs):
        raise NotImplementedError(whoami(self))

    def dir(self, *args, **kwargs):
        self.pull_cache()
        raise NotImplementedError(whoami(self))


class HDF5File(File):
    def __init__(self, _metadata=None, *args, mode="r", **kwargs):
        super().__init__(_metadata, *args,   **kwargs)
        self._root = None

    @property
    def root(self):
        if self._root is None:
            self._root = h5py.File(self.path)
        return HDF5Node(self._root)

    def update(self, d):
        raise NotImplementedError()


class HDF5Collection(CollectionLocalFile):
    def __init__(self, uri, *args, **kwargs):
        super().__init__(*args, file_extension=".h5",   file_factory=None, **kwargs)


__SP_EXPORT__ = HDF5File
