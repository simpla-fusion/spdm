
import collections
import copy
import numbers
from functools import cached_property, lru_cache
import inspect
import types
import numpy as np
import scipy
from scipy import constants
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from spdm.util.AttributeTree import AttributeTree, _last_, _next_
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger
import matplotlib.pyplot as plt
import numpy as np


def make_x_axis(x_axis=None, default_npoints=129):
    if not isinstance(x_axis, np.ndarray) and not x_axis:
        x_axis = default_npoints

    if isinstance(x_axis, int):
        res = np.linspace(0.0, 1.0, x_axis)
    elif isinstance(x_axis, np.ndarray):
        res = x_axis
    elif isinstance(x_axis, collections.abc.Sequence):
        res = np.array(x_axis)
    else:
        raise TypeError(f"Illegal x_axis type! Need 'int' or 'ndarray', not {type(x_axis)}.")

    return res


class Profile(np.ndarray):

    def __new__(cls,  value=None,  x_axis=None,  *args,   **kwargs):
        if cls is not Profile:
            return super(Profile, cls).__new__(cls)
        if isinstance(x_axis, np.ndarray):
            shape = x_axis.shape
        elif isinstance(x_axis, int):
            shape = [x_axis]
        elif isinstance(x_axis, collections.abc.Sequence):
            shape = x_axis
        elif isinstance(value, np.ndarray):
            shape = value.shape
        elif isinstance(value, collections.abc.Sequence):
            shape = [len(value)]
        else:
            shape = [0]

        obj = super(Profile, cls).__new__(cls, shape)
        obj._x_axis = make_x_axis(x_axis)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._x_axis = getattr(obj, '_x_axis', None)
        self._description = {}
        self._unit = None

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        args = []
        in_no = []
        x_axis = None
        for i, input_ in enumerate(inputs):
            if isinstance(input_, Profile):
                if x_axis is None:
                    x_axis = getattr(input_, "_x_axis", None)
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, Profile):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        info = {}
        if in_no:
            info['inputs'] = in_no
        if out_no:
            info['outputs'] = out_no

        # results = super(Profile, self).__array_ufunc__(ufunc, method, *args, **kwargs)

        results = getattr(ufunc, method)(*args, **kwargs)

        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], Profile):
                inputs[0].info = info
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(Profile) if output is None else output)
                        for result, output in zip(results, outputs))
        if results and isinstance(results[0], Profile):
            results[0]._x_axis = x_axis
        return results[0] if len(results) == 1 else results

    def __init__(self,  value=None, x_axis=None,  *args, unit=None, description=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._description = description
        self._unit = unit
        self.assign(value)

    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        if isinstance(res, Profile):
            res._x_axis = self._x_axis[idx]
        return res

    @property
    def description(self):
        return self._description

    @property
    def unit(self):
        return self._unit

    @property
    def x_axis(self):
        return self._x_axis

    def assign(self, other):
        if isinstance(other, Profile) and getattr(other, "_x_axis", None) is not self._x_axis:
            # self._x_axis = other._x_axis
            # self.resize(self._x_axis.size, refcheck=False)
            # self.reshape(self._x_axis.shape)
            np.copyto(self, other(self._x_axis))

        elif isinstance(other, np.ndarray):
            if self.shape != self._x_axis.shape:
                self.resize(self._x_axis.size, refcheck=False)
                self.reshape(self._x_axis.shape)
            np.copyto(self, other)
        elif isinstance(other, (int, float)):
            self.fill(other)
        elif callable(other) or isinstance(other, types.BuiltinFunctionType):
            try:
                v = other(self._x_axis)
            except Exception:
                v = np.array([other(x) for x in self._x_axis])
            finally:
                np.copyto(self, v)

        elif not other:
            self.fill(0)
        else:
            raise ValueError(f"Illegal profiles! {type(other)}")

    @property
    def _interpolate_func(self):
        return UnivariateSpline(self._x_axis, self)

    def __call__(self, *args, **kwargs):
        return self._interpolate_func(*args, **kwargs)

    @property
    def derivative(self):
        return Profile(self._interpolate_func.derivative()(self._x_axis), x_axis=self._x_axis)

    def integral(self,   start=None, stop=None):
        func = self._interpolate_func
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


class Profiles(AttributeTree):
    """ Collection of profiles with same x-axis
    """

    def __init__(self, cache=None, *args,  x_axis=None,  parent=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(cache, LazyProxy) or isinstance(cache, AttributeTree):
            self.__dict__["_cache"] = cache
        else:
            self.__dict__["_cache"] = AttributeTree(cache)

        if isinstance(x_axis, str):
            x_axis = self._cache[x_axis]

        self.__dict__["_x_axis"] = make_x_axis(x_axis)

    def _create(self, d=None, name=None):
        if isinstance(d, Profile) and not hasattr(d, "_x_axis"):
            d._x_axis = self._x_axis
        else:
            d = Profile(d, x_axis=self._x_axis, description={"name": name})
        return d

    def __missing__(self, key):
        d = self._cache[key]
        if isinstance(d, LazyProxy):
            d = d()
        if d in (None, [], {}, NotImplemented) or len(d) == 0:
            return None
        else:
            return self.__as_object__().setdefault(key, self._create(d, name=key))

    def __setitem__(self, key, value):
        if isinstance(value, Profile) and hasattr(value, "_x_axis"):
            self.__as_object__()[key] = value
            return
        v = self.__getitem__(key)
        if v is None:
            self.__as_object__()[key] = self._create(value, name=key)
        elif isinstance(v, Profile):
            v.assign(value)
        elif isinstance(v, np.ndarray):
            v[:] = value[:]
        else:
            raise KeyError(f"Can not assign value to {key}: {type(v)}!")

    @lru_cache
    def cache(self, key):
        res = self._cache[key.split(".")]
        if isinstance(res, LazyProxy):
            res = res()
        return res

    def _as_profile(self, y, new_axis=None, **kwargs):
        if not isinstance(y, Profile) or new_axis is not None:
            y = Profile(y, x_axis=new_axis, **kwargs)

        if not hasattr(y, "_x_axis"):
            y._x_axis = self._x_axis

        return y

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
        return self._as_profile(self._interpolate_item(key).derivative())

    def derivative_func(self, func, x_axis=None,  **kwargs):
        if isinstance(func, str) and x_axis is None:
            res = self._derivative(func)
        elif x_axis is None:
            res = self.interpolate(func).derivative(**kwargs)
        else:
            res = self.mapping(x_axis, func).derivative(**kwargs)
        return self._as_profile(res)

    def derivative(self, *args, **kwargs):
        return self.derivative_func(*args, **kwargs)(self._x_axis)

    def dln(self, *args, **kwargs):
        r"""
            .. math:: d\ln f=\frac{df}{f}
        """
        return self.derivative/self

    def mapping(self, x_axis, y, new_axis=None):
        if isinstance(x_axis, str):
            x_axis = self.__getitem__(x_axis)

        if isinstance(y, str):
            y = self.__getitem__(y)

        if not isinstance(y, np.ndarray):
            return None
        elif y.shape != x_axis.shape:
            raise RuntimeError(f"x,y length is not same! { x_axis.shape }!={y.shape} ")

        return self._as_profile(y, new_axis)

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
