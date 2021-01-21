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
    def __init__(self, holder,  *args, envs=None, time=None, time_slice=None, mode=None, **kwargs):
        super().__init__(holder, *args, **kwargs)
        self._time = time
        self._time_slice = time_slice
        self._envs = envs or {}
        self._mode = mode or "r"

        if "r" in self.mode and self._+holder.isConnected():
            if self._time_slice is None:
                self._holder.get()
            else:
                self._holder.getSlice(self._time_slice)
        else:
            self._holder.ids_properties.homogeneous_time = self._homogeneous_time
            self._holder.ids_properties.creation_date = self._creation_date
            self._holder.ids_properties.provider = self._provider
            self._holder.time = self._time

    def _get_ids(self, obj, path):
        assert(isinstance(path, list))
        if path == [None]:
            return obj

        for idx, p in enumerate(path):
            t_obj = None
            if obj.__class__.__name__.endswith("__structArray"):
                if isinstance(p, str):
                    logger.debug((type(obj), obj._base_path))
                    t_obj = getattr(self._get_ids(obj, [self._time_slice]), p, None)
                elif isinstance(p, (int, slice)):
                    try:
                        t_obj = obj[p]
                    except IndexError as error:
                        if hasattr(obj.__class__, 'resize') and ("w" in self._mode or "x" in self._mode):
                            logger.debug(f"Resize object {obj.__class__} {len(obj)} {p+1}")
                            obj.resize(p+1)
                            t_obj = obj[p]
                        else:
                            t_obj = None
            elif isinstance(p, str):
                t_obj = getattr(obj, p, None)

            if t_obj is None:
                logger.error((type(obj),   path))
                raise KeyError('.'.join(map(str, path[:idx+1])))

            obj = t_obj

        return obj

    def _put_value(self, obj, key, value):

        if isinstance(value, (int, float, str)):
            if isinstance(key, str):
                setattr(obj, key, value)
            else:
                obj[key] = value
        elif isinstance(value, np.ndarray):
            self._get_ids(obj, [key, "resize"])(*value.shape)
            self._get_ids(obj, [key])[:] = value[:]
        elif isinstance(value, list):
            t_obj = self._get_ids(obj, [key])
            if len(t_obj) < len(value):
                self._get_ids(obj, [key, "resize"])(len(value))
                obj = self._get_ids(obj, [key])
            else:
                obj = t_obj
            for idx, v in enumerate(value):
                self._put_value(obj, idx, value)

        elif isinstance(value, collections.abc.Mapping):
            obj = self._get_ids(obj, [key])
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
            self._put_value(self._get_ids(self._holder, path), None, value)
        else:
            self._put_value(self._get_ids(self._holder, path[:-1]), path[-1], value)

    # def _get(self, path):
    #     if not path:
    #         return self._holder

    #     # assert(len(path) > 0)
    #     obj=self._holder
    #     if obj.__class__.__name__.endswith('__structArray'):
    #         if isinstance(p[0],str):
    #             obj=self._get_ids(obj,0)

    #     obj = getattr(self._holder, path[0], None)

    #     if obj is None:
    #         raise KeyError(f"{type(self._holder)} {path[0]}")

    #     # only structArray in first level is time dependent
    #     logger.debug(obj.__class__)
    #     if obj.__class__.__name__.endswith('__structArray'):
    #         time_slice_length = len(self._time) if isinstance(self._time, np.ndarray) else 1
    #         if len(obj) < time_slice_length:
    #             obj.resize(time_slice_length)
    #         if len(path) > 1 and isinstance(path[1], str):
    #             obj = obj[self._time_slice]

    #     return self._get_ids(obj, path[1:])

    def _wrap_obj(self, res):
        if isinstance(res, (int, float, list, dict, AttributeTree, np.ndarray)):
            return res
        else:
            logger.debug(type(res))
            return IMASNode(res, envs=self._envs, time=self._time, time_slice=self._time_slice)

    def get(self, path):

        return self._wrap_obj(self._get_ids(self._holder, path))

    def iter(self,   path, *args, **kwargs):
        obj = self._get_ids(self._holder, path)
        if obj.__class__.__name__.endswith("__structArray"):
            for idx in range(len(obj)):
                yield self._wrap_obj(self._get_ids(obj, [idx]))
        else:
            yield from self._wrap_obj(obj)
            # raise NotImplementedError()


class IMASDocument(Document):
    def __init__(self,  *args, shot=0, run=0, data=None, time=0.0, time_slice=None, **kwargs):
        super().__init__(*args,  ** kwargs)

        self._entry = None

        self._time_slice = time_slice

        self._data = data or imas.ids(int(shot), int(run))

        self._cache = {}

        logger.info(
            f"Open IMAS Document: shot={self._data.getShot()} run={self._data.getRun()} isConnected={self._data.isConnected()}")

        if "x" in self.mode or "w" in self.mode:
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
        else:
            self._time = None
            self._homogeneous_time = None

        # if "x" in self.mode or "w" in self.mode:
        #     self._creation_date = datetime.datetime.now().ctime()
        #     self._provider = os.environ.get('USER', 'nobody')

    def close(self):
        self.flush()
        if self._data is not None and self._data.isConnected():
            self._data.close()
        logger.info(f"Close IMAS Document")

        super().close()

    def get_ids(self,  p):
        assert(isinstance(p, str))

        ids = self._cache.get(p, None) or self._cache.setdefault(p, getattr(self._data, p, None))

        if not ids:
            raise KeyError(f"Can not get ids '{p}'!")

        return IMASNode(ids, mode=self.mode, envs=self.envs, time=self._time, time_slice=self._time_slice)

    def flush(self):
        if "w" in self.mode or "x" in self.mode:
            for ids in self._access_cache:
                ids.put()
        self._cache.clear()

    @property
    def entry(self):
        if self._entry is None:
            self._entry = LazyProxy(None,
                                    get=lambda o, p, _s=self: _s.get_ids(p[0]).get(p[1:]),
                                    get_value=lambda o, p, _s=self: _s.get_ids(p[0]).get(p[1:]),
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
    def __init__(self, uri, *args,  user=None,  database=None, version=None, local_path=None, mode=None,   **kwargs):
        super().__init__(uri, *args, id_hasher="{shot}_{run}", ** kwargs)

        self._user = user or os.environ.get("USER", "nobody")

        self._mode = mode or "rw"

        o = urisplit(uri)

        if not not o.authority:
            raise NotImplementedError(o.authority)

        self._database = database or o.query.device or "unnamed"

        self._version = version or o.query.version or os.environ.get('IMAS_VERSION', '3').split('.', 1)[0]

        self._path = f"{local_path or o.path or '~/public/imasdb'}/{self._database}/{self._version}"

        if self._path[:2] == '/~':
            self._path = self._path[1:]

        self._path = pathlib.Path(self._path).expanduser().resolve()

        default_path = pathlib.Path(f"~/public/imasdb/{self._database}/{self._version}").expanduser().resolve()

        if self._path != default_path:
            raise NotImplementedError(f"Can not change imasdb path  from '{default_path}'' to '{self._path}'!")

        logger.info(f"Set IMAS database path: {self._path}")

        for i in range(10):
            p = self._path/f"{i}"
            p.mkdir(parents=True, exist_ok=True)
            os.environ[f"MDSPLUS_TREE_BASE_{i}"] = p.as_posix()

    def open_document(self, fid=None, *args, mode=None, shot=None, run=None, time_slice=None, **kwargs):

        if fid is not None:
            shot, run = fid.split('_')

        shot = int(shot or 0)

        run = int(run or 0)

        ids = imas.ids(shot, run)

        mode = mode or self._mode

        status = None
        idx = None
        # logger.debug(os.environ[f"MDSPLUS_TREE_BASE_0"])
        if "x" in mode:
            status, idx = ids.create_env(self._user, self._database, self._version)
        else:
            status, idx = ids.open_env(self._user, self._database, self._version)

        if status != 0:
            raise RuntimeError(
                f"Can not open imas databas user={self._user} database={self._database} version={self._version} error status={status}")
        return IMASDocument(data=ids,  mode=mode, **kwargs)


__SP_EXPORT__ = IMASCollection
