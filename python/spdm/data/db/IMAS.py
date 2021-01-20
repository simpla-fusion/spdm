import collections
import datetime
import os
import pathlib
from functools import lru_cache

import imas
import numpy as np
from spdm.util.AttributeTree import AttributeTree
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
from spdm.util.PathTraverser import PathTraverser
from spdm.util.urilib import urisplit

from ..Collection import Collection
from ..Document import Document
from ..Node import Node


class IMASNode(Node):
    def __init__(self, holder,  *args, envs=None, time=None, **kwargs):
        super().__init__(holder, *args, **kwargs)
        self._time = time
        self._envs = envs or {}

    def _get_ids(self, obj, path):
        if path is None:
            return obj
        if isinstance(path, str):
            path = path.split('/')
        elif isinstance(path, (int, slice)):
            path = [path]

        if not path:
            return obj

        prev = None

        for idx, p in enumerate(path):
            if obj.__class__.__name__.endswith("__structArray"):
                if isinstance(p, str):
                    t_obj = getattr(self._get_ids(obj, 0), p, None)
                elif isinstance(p, (int, slice)):
                    try:
                        t_obj = obj[p]
                    except IndexError as error:
                        if hasattr(obj.__class__, 'resize'):
                            logger.debug(f"Resize object {obj.__class__} {len(obj)} {p+1}")
                            obj.resize(p+1)
                            t_obj = obj[p]
                        else:
                            t_obj = None
            elif isinstance(p, str):
                t_obj = getattr(obj, p, None)
            else:
                t_obj = None

            if t_obj is None:
                logger.error((type(obj), '.'.join(map(str, path[:idx+1]))))
                raise KeyError('.'.join(map(str, path[:idx+1])))
            else:
                obj = t_obj

            prev = p
        return obj

    def _put_value(self, obj, key, value):

        if isinstance(value, (int, float, str)):
            if isinstance(key, str):
                setattr(obj, key, value)
            else:
                obj[key] = value
        elif isinstance(value, np.ndarray):
            self._get_ids(obj, [key, "resize"])(*value.shape)
            self._get_ids(obj, key)[:] = value[:]
        elif isinstance(value, list):
            t_obj = self._get_ids(obj, key)
            if len(t_obj) < len(value):
                self._get_ids(obj, [key, "resize"])(len(value))
                obj = self._get_ids(obj, key)
            else:
                obj = t_obj
            for idx, v in enumerate(value):
                self._put_value(obj, idx, value)

        elif isinstance(value, collections.abc.Mapping):
            obj = self._get_ids(obj, key)
            for k, v in value.items():
                self._put_value(obj, k, v)
        else:
            raise NotImplementedError(type(value))

    def put(self, path, value, *args, **kwargs):
        if isinstance(value, LazyProxy):
            value = value.__value__()
        if isinstance(value, AttributeTree):
            value = value.__as_native__()

        if len(path) == 0 and isinstance(value, collections.abc.Mapping):
            for k, v in value.items():
                self.put([k], v)
        elif len(path) == 0:
            raise ValueError(type(value))
        elif len(path) == 1:
            self._put_value(self.get(path), None, value)
        else:
            self._put_value(self.get(path[:-1]), path[-1], value)

    def get(self, path):

        assert(len(path) > 0)
        obj = getattr(self._holder, path[0], None)
        if obj is None:
            raise KeyError(f"{type(self._holder)} {path[0]}")

        # only structArray in first level is time dependent
        if obj.__class__.__name__.endswith('__structArray'):
            time_slice_length = len(self._time) if isinstance(self._time, np.ndarray) else 1
            if len(obj) < time_slice_length:
                obj.resize(time_slice_length)

        return self._get_ids(obj, path[1:])

    def iter(self, holder, path, *args, **kwargs):
        raise NotImplementedError()


class IMASDocument(Document):
    def __init__(self,  *args, shot=0, run=0, data=None, time=0.0, **kwargs):
        super().__init__(*args,  ** kwargs)

        self._data = data or imas.ids(int(shot), int(run))

        self._entry = None

        if isinstance(time, list):
            self._time = np.array(time)
        elif isinstance(time, np.ndarray):
            self._time = time
        elif time is not None:
            self._time = float(time)

        if self._time is None:
            self._homogeneous_time = 0
        elif isinstance(self._time, np.ndarray):
            self._homogeneous_time = 1
        elif isinstance(self._time, float):
            self._homogeneous_time = 2

        self._creation_date = datetime.datetime.now().ctime()
        self._provider = os.environ.get('USER', 'nobody')

        logger.info(f"Open IMAS Document: {'connected' if self._data.isConnected() else 'Single'}")

    def close(self):
        if self._data is not None and self._data.isConnected():
            self._data.close()
        logger.info(f"Close IMAS Document")

        super().close()

    @lru_cache
    def get_ids(self,  p):
        assert(isinstance(p, str))

        ids = getattr(self._data, p, None)

        if not ids:
            raise KeyError(f"Can not get ids '{p}'!")

        ids.ids_properties.homogeneous_time = self._homogeneous_time
        ids.time = self._time
        ids.creation_date = self._creation_date
        ids.provider = self._provider

        return IMASNode(ids, mode=self.mode, envs=self.envs, time=self._time)

    @property
    def entry(self):
        if self._entry is None:
            self._entry = LazyProxy(None,
                                    get=lambda o, p, _s=self: _s.get_ids(p[0]).get(p[1:]),
                                    put=lambda o, p, v, _s=self: _s.get_ids(p[0]).put(p[1:], v),
                                    )
        return self._entry

    def update(self, d):
        if isinstance(d, Document):
            d = d.entry
        if isinstance(d, LazyProxy):
            d = d.__real_value__()
        if isinstance(d, AttributeTree):
            d = d.__as_native__()

        for k, v in d.items():
            self.get_ids(k).put([], v)

        # IMASNode(self._data, mode=self.mode, envs=self.envs).put([], d)


class IMASCollection(Collection):
    def __init__(self, uri, *args,  user=None,  database=None, version=None,   **kwargs):
        super().__init__(uri, *args, id_hasher="{shot}_{run}", ** kwargs)

        self._user = user or os.environ.get("USER", "NOBODY")

        o = urisplit(uri)

        self._database = database or o.authority

        self._version = version or os.environ.get('IMAS_VERSION', '3').split('.', 1)[0]

        self._local_db = pathlib.Path(f"~/public/imasdb/{self._database}/{self._version}/0").expanduser().resolve()

        self._local_db.mkdir(parents=True, exist_ok=True)

        self._data.create_env(user, database, version)

    def open_document(self, fid=None, *args, mode=None, **kwargs):
        shot, run = fid.split('_')
        return IMASDocument(*args,  mode=mode, user=self._user,
                            database=self._database, shot=shot, run=run,
                            envs=collections.ChainMap(kwargs, self.envs))

        # user = user or os.environ.get("USER", "NOBODY")

        # database = database or 'UNNAMED_DB'

        # version = version or os.environ.get('IMAS_VERSION', '3').split('.', 1)[0]

        # if "r" in self.mode:
        #     self._data.open_env(user, database, version)


__SP_EXPORT__ = IMASCollection
