
import collections
import copy
import numbers
from functools import cached_property, lru_cache

import numpy as np
import scipy
from scipy import constants
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from spdm.util.AttributeTree import AttributeTree, _last_, _next_
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
import matplotlib.pyplot as plt
import numpy as np


class Profiles(AttributeTree):
    """ Collection of profiles with same x-axis
    """

    def __init__(self, cache, *args, x_axis=129, default_npoints=129, parent=None, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(cache, LazyProxy) or isinstance(cache, AttributeTree):
            pass
        else:
            cache = AttributeTree(cache)

        self.__dict__["_cache"] = cache

        if isinstance(x_axis, str):
            x_axis = self.cache(x_axis)

        if type(x_axis) is int:
            x_axis = np.linspace(0.0, 1.0, x_axis)

        if isinstance(x_axis, np.ndarray) and len(x_axis) > 0:
            pass
        else:
            x_axis = np.linspace(0.0, 1.0, default_npoints)

        self.__dict__["_x_axis"] = x_axis

    def __missing__(self, key):
        d = self._cache[key]
        if isinstance(d, LazyProxy):
            d = d()

        if isinstance(d, np.ndarray):
            pass
        elif callable(d):
            d = d(self._x_axis)
        elif isinstance(d, numbers.Number):
            d = np.full(self._x_axis.shape, float(d))
        elif d is (NotImplemented, None, [], {}):
            d = np.zeros(self._x_axis.shape)

        return d

    def __setitem__(self, key, value):
        if isinstance(value, np.ndarray) and value.shape == self._x_axis.shape:
            pass
        elif callable(value):
            value = np.array([value(x) for x in self._x_axis])
        elif type(value) in (int, float):
            value = np.full(self._x_axis.shape, value)
        else:
            raise TypeError(f"{type(value)}")
        super().__setitem__(key, value)

    @lru_cache
    def cache(self, key):
        res = self._cache[key.split(".")]
        if isinstance(res, LazyProxy):
            res = res()
        return res

    @lru_cache
    def _interpolate_item(self, key):
        y = self.__getitem__(key)
        if not isinstance(y, np.ndarray) and y in (None, [], {}):
            raise LookupError(key)
        return self.interpolate(y)

    def interpolate(self, func, **kwargs):
        if isinstance(func, str):
            return self._interpolate_item(func)
        elif isinstance(func, np.ndarray) and func.shape[0] == self._x_axis.shape[0]:
            return UnivariateSpline(self._x_axis, func, **kwargs)
        elif isinstance(func, collections.abc.Sequence):
            return {k: self.interpolate(k, **kwargs) for k in func}
        elif callable(func):
            return func
        else:
            raise ValueError(f"Cannot create interploate! {func}")

    def integral(self, func, start=None, stop=None):
        func = self.interpolate(func)
        if start is None:
            start = self._x_axis[0]
        if stop is None:
            stop = self._x_axis[-1]

        if not isinstance(start, np.ndarray) and not isinstance(stop, np.ndarray):
            return func.integral(start, stop)
        elif (isinstance(start, np.ndarray) or isinstance(start, collections.abc.Sequence)):
            return np.array([func.integral(x, stop) for x in start])
        elif (isinstance(stop, np.ndarray) or isinstance(stop, collections.abc.Sequence)):
            return np.array([func.integral(start, x) for x in stop])
        else:
            raise TypeError(f"{type(start)},{type(stop)}")

    @lru_cache
    def _derivative(self, key):
        return self._interpolate_item(key).derivative()(self._x_axis)

    def derivative(self, func,   **kwargs):
        if isinstance(func, str):
            return self._derivative(func)
        elif isinstance(func, collections.abc.Sequence):
            return {k: self.derivative(k, **kwargs) for k in func}
        else:
            return self.interpolate(func).derivative(**kwargs)(self._x_axis)

    @lru_cache
    def mapping(self, x_axis, key):
        if isinstance(x_axis, str):
            x_axis = self.__getitem__(x_axis)
        y = self.__getitem__(key)
        if not isinstance(y, np.ndarray):
            raise LookupError(f"'{key}'")

        return UnivariateSpline(x_axis, y)

    def _fetch_profile(self, desc, prefix=[]):
        if isinstance(desc, str):
            path = desc
            opts = {"label": desc}
        elif isinstance(desc, collections.abc.Mapping):
            path = desc.get("name", None)
            opts = desc.get("opts", {})
        elif isinstance(desc, tuple):
            path, opts = desc
        elif isinstance(desc, AttributeTree):
            path = desc.data
            opts = desc.opts
        else:
            raise TypeError(f"Illegal profile type! {desc}")

        if isinstance(opts, str):
            opts = {"label": opts}

        if prefix is None:
            prefix = []
        elif isinstance(prefix, str):
            prefix = prefix.split(".")

        if isinstance(path, str):
            path = path.split(".")

        path = prefix+path

        if isinstance(path, np.ndarray):
            data = path
        else:
            data = self[path]

        # else:
        #     raise TypeError(f"Illegal data type! {prefix} {type(data)}")

        return data, opts

    def plot(self, profiles, axis=None, x_axis=None, prefix=None):
        if isinstance(profiles, str):
            profiles = profiles.split(",")
        elif not isinstance(profiles, collections.abc.Sequence):
            profiles = [profiles]

        if prefix is None:
            prefix = []
        elif isinstance(prefix, str):
            prefix = prefix.split(".")
        elif not isinstance(prefix, collections.abc.Sequence):
            prefix = [prefix]

        x_axis, x_axis_opts = self._fetch_profile(x_axis, prefix=prefix)

        fig = None
        if isinstance(axis, collections.abc.Sequence):
            pass
        elif axis is None:
            fig, axis = plt.subplots(ncols=1, nrows=len(profiles), sharex=True)
        elif len(profiles) == 1:
            axis = [axis]
        else:
            raise RuntimeError(f"Too much profiles!")

        for idx, data in enumerate(profiles):
            ylabel = None
            opts = {}
            if isinstance(data, tuple):
                data, ylabel = data

            if not isinstance(data, list):
                data = [data]

            for d in data:
                value, opts = self._fetch_profile(d,  prefix=prefix)

                if value is not NotImplemented and value is not None and len(value) > 0:
                    axis[idx].plot(x_axis, value, **opts)
                else:
                    logger.error(f"Can not find profile '{d}'")

            axis[idx].legend(fontsize=6)

            if ylabel:
                axis[idx].set_ylabel(ylabel, fontsize=6).set_rotation(0)
            axis[idx].labelsize = "media"
            axis[idx].tick_params(labelsize=6)

        axis[-1].set_xlabel(x_axis_opts.get("label", ""),  fontsize=6)

        return axis, fig
