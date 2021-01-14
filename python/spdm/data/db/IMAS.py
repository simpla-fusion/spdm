import collections
import os
import pathlib
import numpy as np
import imas
from spdm.util.urilib import urisplit
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.PathTraverser import PathTraverser

from ..Collection import Collection
from ..Document import Document
from ..Node import Node


class IMASNode(Node):
    def __init__(self, holder,  *args, envs=None, **kwargs):
        super().__init__(holder, *args, **kwargs)
        self._envs = envs or {}

    def _get_ids(self, obj, path, time_slice=None):
        if not path:
            return obj
        if isinstance(path, str):
            path = path.split('/')
        prev = None

        for idx, p in enumerate(path):
            if prev == "time_slice" and not isinstance(p, (int, slice)) and p != "resize":
                time_slice = time_slice or self.envs.get("time_slice", 0)
                t_obj = obj[time_slice]
            if isinstance(p, str):
                t_obj = getattr(obj, p, None)
            elif isinstance(p, (int, slice)):
                t_obj = obj[p]
            else:
                t_obj = None

            if t_obj is None:
                raise KeyError(path)  # '.'.join(path[:idx+1]))
            else:
                obj = t_obj

            prev = p
        return obj

    def _put_ids(self, obj, path, value, *args, **kwargs):
        if not path:
            if isinstance(value, list):
                obj.resize(len(value))
                for idx, v in enumerate(value):
                    self._put_ids(obj, idx, v)
            elif isinstance(value, collections.abc.Mapping):
                obj.resize(len(value))
                for k, v in value.items():
                    self._put_ids(obj, k, v)
        elif isinstance(value, (int, float, str)):
            if isinstance(path, str):
                path = path.split('/')
            obj = self._get_ids(obj, path[:-1])
            key = path[-1]
            if isinstance(key, str):
                setattr(obj, key, value)
            else:
                obj[key] = value
        elif isinstance(value, np.ndarray):
            obj = self._get_ids(obj, path[:-1])
            self._get_ids(obj, [path[-1], "resize"])(*value.shape)
            aobj = self._get_ids(obj, path[-1])
            aobj[:] = value[:]

        else:
            self._put_ids(self._get_ids(obj, path), [], value)

    def _fix_time_slice(self, path):
        if len(path) >= 3 and path[1] == 'time_slice' and isinstance(path[2], str):
            time_slice = self._envs.get("time_slice", 0)
            path = path[:2]+[time_slice]+path[2:]
        return path

    def put(self, path, value, *args, **kwargs):
        path = self._fix_time_slice(path)
        self._put_ids(self.holder, path, value)
        if len(path) > 1 and path[1] == "time":
            ids = getattr(self.holder, path[0])
            if isinstance(value, float):
                ids.time_slice.resize(1)
                ids.ids_properties.homogeneous_time = 2
            elif isinstance(value, np.ndarray):
                ids.ids_properties.homogeneous_time = 1
                ids.time_slice.resize(value.size)
            else:
                ids.ids_properties.homogeneous_time = 0
            # getattr(self.holder, path[0]).putSlice()
            # getattr(self.holder, path[0]).put()

    def get(self, path, *args,    **kwargs):
        path = self._fix_time_slice(path)
        return self._get_ids(self._holder, path, **kwargs)

    def iter(self, holder, path, *args, **kwargs):
        raise NotImplementedError()


class IMASDocument(Document):
    def __init__(self,  *args, shot=0, run=0, database=None, user=None, version=None, **kwargs):
        super().__init__(*args,  ** kwargs)

        user = user or os.environ.get("USER", "NOBODY")

        database = database or 'UNNAMED_DB'

        version = version or os.environ.get('IMAS_VERSION', '3').split('.', 1)[0]

        self._data = imas.ids(int(shot), int(run))

        if "r" in self.mode:
            self._data.open_env(user, database, version)
        else:
            self._data.create_env(user, database, version)

        logger.info(f"Open IMAS Document {user, database, version}: {'OK' if self._data.isConnected() else 'FAILED'}")

    def close(self):
        if not not self._data:
            logger.info(f"Close IMAS Document")
            self._data.equilibrium.put()
            logger.debug(len(self._data.equilibrium.time_slice))
            self._data.close()
        super().close()

    @property
    def root(self):
        return IMASNode(self._data, mode=self.mode, envs=self.envs)

    def update(self, d):
        IMASNode(self._data, mode=self.mode, envs=self.envs).put(d)


class IMASCollection(Collection):
    def __init__(self, uri, *args,  user=None,  database=None, version=None,   **kwargs):
        super().__init__(uri, *args, id_hasher="{shot}_{run}", ** kwargs)

        self._user = user or os.environ.get("USER", "NOBODY")

        o = urisplit(uri)

        self._database = database or o.authority

        self._version = version or os.environ.get('IMAS_VERSION', '3').split('.', 1)[0]

        self._local_db = pathlib.Path(f"~/public/imasdb/{self._database}/{self._version}/0").expanduser().resolve()

        self._local_db.mkdir(parents=True, exist_ok=True)

    def open_document(self, fid=None, *args, mode=None, **kwargs):
        shot, run = fid.split('_')
        return IMASDocument(*args,  mode=mode, user=self._user,
                            database=self._database, shot=shot, run=run,
                            envs=collections.ChainMap(kwargs, self.envs))


__SP_EXPORT__ = IMASCollection
