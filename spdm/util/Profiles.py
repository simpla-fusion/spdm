
import collections
import copy
import inspect
import numbers
import types
from functools import cached_property, lru_cache

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.misc
from scipy import constants
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from spdm.util.AttributeTree import AttributeTree, _last_, _next_
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger


def make_axis(axis=None, default_npoints=129):
    if not isinstance(axis, np.ndarray) and not axis:
        axis = default_npoints

    if isinstance(axis, int):
        res = np.linspace(0.0, 1.0, axis)
    elif isinstance(axis, np.ndarray):
        res = axis
    elif isinstance(axis, collections.abc.Sequence):
        res = np.array(axis)
    else:
        raise TypeError(f"Illegal axis type! Need 'int' or 'ndarray', not {type(axis)}.")

    return res


class Profile(np.ndarray):

    def __new__(cls,  value=None,  axis=None,  *args,   **kwargs):
        if cls is not Profile:
            return super(Profile, cls).__new__(cls)
        if isinstance(axis, np.ndarray):
            shape = axis.shape
        elif isinstance(axis, int):
            shape = [axis]
        elif isinstance(axis, collections.abc.Sequence):
            shape = axis
        elif isinstance(value, np.ndarray):
            shape = value.shape
        elif isinstance(value, collections.abc.Sequence):
            shape = [len(value)]
        else:
            shape = [0]

        obj = super(Profile, cls).__new__(cls, shape)
        obj._axis = make_axis(axis)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._axis = getattr(obj, '_axis', None)
        self._description = {}
        self._unit = None

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        # logger.debug((ufunc, method))
        args = []
        in_no = []
        axis = None
        for i, input_ in enumerate(inputs):
            if isinstance(input_, Profile):
                in_no.append(i)
                if axis is None:
                    axis = getattr(input_, "axis", None)

                # or (input_.axis.shape == axis.shape and all(input_.axis == axis)):
                if input_.axis is axis:
                    args.append(input_.view(np.ndarray))
                else:
                    data = input_(axis)
                    args.append(data.view(np.ndarray))

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
            results[0]._axis = axis
        return results[0] if len(results) == 1 else results

    def __init__(self,  value=None, axis=None,  *args, description=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._description = description

        self.assign(value)

    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        if isinstance(res, Profile):
            res._axis = self._axis[idx]
        return res

    @property
    def description(self):
        return self._description

    @property
    def unit(self):
        return self._description.get("unit", None)

    @property
    def axis(self):
        return self._axis

    def assign(self, other):
        if isinstance(other, Profile) and getattr(other, "_axis", None) is not self._axis:
            # self._axis = other._axis
            # self.resize(self._axis.size, refcheck=False)
            # self.reshape(self._axis.shape)
            np.copyto(self, other(self._axis))

        elif isinstance(other, np.ndarray):
            if self.shape != self._axis.shape:
                self.resize(self._axis.size, refcheck=False)
                self.reshape(self._axis.shape)
            np.copyto(self, other)
        elif isinstance(other, (int, float)):
            self.fill(other)
        elif callable(other) or isinstance(other, types.BuiltinFunctionType):
            self._func = other
            try:
                v = other(self._axis)
            except Exception:
                v = np.array([other(x) for x in self._axis])
            finally:
                np.copyto(self, v)

        elif not other:
            self.fill(0)
        else:
            raise ValueError(f"Illegal profiles! {type(other)}")

    @property
    def interpolate(self):
        return UnivariateSpline(self._axis, self.view(np.ndarray))

    def __call__(self, axis, *args, **kwargs):
        if axis is not self._axis:
            pass
        elif isinstance(axis, np.ndarray) and all(axis.view(np.ndarray) == self._axis.view(np.ndarray)):
            return self

        func = getattr(self, "_func", None) or self.interpolate

        res = func(axis, *args, **kwargs)

        if not isinstance(res, np.ndarray) or len(res.shape) == 0:
            pass
        elif len(res.shape) == 1 and res.size == 1:
            res = res[0]
        else:
            res = Profile(res, axis=axis)

        return res

    def derivative_n(self, n, *args, **kwargs):
        func = getattr(self, "_func", None)
        if func is None:
            return Profile(self.interpolate.derivative(n=n)(self._axis), axis=self._axis)
        elif callable(func) or isinstance(func, types.BuiltinFunctionType):
            v0 = scipy.misc.derivative(func, self._axis[0], dx=self._axis[1]-self._axis[0], n=n, **kwargs)
            vn = scipy.misc.derivative(func, self._axis[-1], dx=self._axis[-1]-self._axis[-2], n=n, **kwargs)
            v = [scipy.misc.derivative(func, x, dx=0.5*(self._axis[i+1]-self._axis[i-1]),
                                       n=n, *args, **kwargs) for i, x in enumerate(self._axis[1:-1])]
            return np.array([v0]+v+[vn])

    @property
    def derivative(self):
        return self.derivative_n(1)

    def integral(self, start=None, stop=None):

        if start is None:
            start = self._axis

        if stop is None:
            stop = self._axis

        func = getattr(self, "_func", None)

        if func is None:
            spl = self.interpolate
            if isinstance(start, np.ndarray) and isinstance(stop, np.ndarray):
                raise ValueError(f"Illegal arguments! start is {type(start)} ,end is {type(end)}")
            elif isinstance(start, np.ndarray):
                res = Profile(np.array([spl.integral(x, stop) for x in start]), axis=start)
            elif isinstance(stop, np.ndarray):
                res = Profile(np.array([spl.integral(start, x) for x in stop]), axis=stop)
            else:
                res = spl.integral(start, stop)
        else:
            if isinstance(start, np.ndarray) and isinstance(stop, np.ndarray):
                raise ValueError(f"Illegal arguments! start is {type(start)} ,end is {type(end)}")
            elif isinstance(start, np.ndarray):
                res = Profile(np.array([scipy.integrate.quad(func, x, stop) for x in start]), axis=start)
            elif isinstance(stop, np.ndarray):
                res = Profile(np.array([scipy.integrate.quad(func, start, x) for x in start]), axis=stop)
            else:
                res = scipy.integrate.quad(func, start, stop)
        return res

    def dln(self, *args, **kwargs):
        r"""
            .. math:: d\ln f=\frac{df}{f}
        """
        return self.derivative/self


class Profiles(AttributeTree):
    """ Collection of profiles with same x-axis
    """

    def __init__(self, cache=None, *args,  axis=None,  parent=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(cache, LazyProxy) or isinstance(cache, AttributeTree):
            self.__dict__["_cache"] = cache
        else:
            self.__dict__["_cache"] = AttributeTree(cache)

        if isinstance(axis, str):
            axis = self._cache[axis]

        self.__dict__["_axis"] = make_axis(axis)

    def _create(self, d=None, name=None):
        if isinstance(d, Profile) and not hasattr(d, "_axis"):
            d._axis = self._axis
        else:
            d = Profile(d, axis=self._axis, description={"name": name})
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
        if isinstance(value, Profile) and hasattr(value, "_axis"):
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

    # @lru_cache
    # def _interpolate_item(self, key):
    #     y = self.__getitem__(key)
    #     if not isinstance(y, np.ndarray) and y in (None, [], {}):
    #         raise LookupError(key)
    #     return self.interpolate(y)

    # def interpolate(self, func, **kwargs):
    #     if isinstance(func, str):
    #         return self._interpolate_item(func)
    #     elif isinstance(func, np.ndarray) and func.shape[0] == self._axis.shape[0]:
    #         return UnivariateSpline(self._axis, func, **kwargs)
    #     elif isinstance(func, collections.abc.Sequence):
    #         return {k: self.interpolate(k, **kwargs) for k in func}
    #     elif callable(func):
    #         return func
    #     else:
    #         raise ValueError(f"Cannot create interploate! {func}")

    # def integral(self, func, start=None, stop=None):
    #     func = self.interpolate(func)
    #     if start is None:
    #         start = self._axis[0]
    #     if stop is None:
    #         stop = self._axis[-1]

    #     if not isinstance(start, np.ndarray) and not isinstance(stop, np.ndarray):
    #         return func.integral(start, stop)
    #     elif (isinstance(start, np.ndarray) or isinstance(start, collections.abc.Sequence)):
    #         return np.array([func.integral(x, stop) for x in start])
    #     elif (isinstance(stop, np.ndarray) or isinstance(stop, collections.abc.Sequence)):
    #         return np.array([func.integral(start, x) for x in stop])
    #     else:
    #         raise TypeError(f"{type(start)},{type(stop)}")

    # @lru_cache
    # def _derivative(self, key):
    #     return self._as_profile(self._interpolate_item(key).derivative())

    # def derivative_func(self, func, axis=None,  **kwargs):
    #     if isinstance(func, str) and axis is None:
    #         res = self._derivative(func)
    #     elif axis is None:
    #         res = self.interpolate(func).derivative(**kwargs)
    #     else:
    #         res = self.mapping(axis, func).derivative(**kwargs)
    #     return self._as_profile(res)

    # def derivative(self, *args, **kwargs):
    #     return self.derivative_func(*args, **kwargs)(self._axis)

    # def dln(self, *args, **kwargs):
    #     r"""
    #         .. math:: d\ln f=\frac{df}{f}
    #     """
    #     return self.derivative/self

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

    def plot(self, profiles, fig_axis=None, axis=None,  prefix=None):
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

        axis, axis_opts = self._fetch_profile(axis, prefix=prefix)

        fig = None
        if isinstance(fig_axis, collections.abc.Sequence):
            pass
        elif fig_axis is None:
            fig, fig_axis = plt.subplots(ncols=1, nrows=len(profiles), sharex=True)
        elif len(profiles) == 1:
            fig_axis = [fig_axis]
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
                    fig_axis[idx].plot(axis, value, **opts)
                else:
                    logger.error(f"Can not find profile '{d}'")

            fig_axis[idx].legend(fontsize=6)

            if ylabel:
                fig_axis[idx].set_ylabel(ylabel, fontsize=6).set_rotation(0)
            fig_axis[idx].labelsize = "media"
            fig_axis[idx].tick_params(labelsize=6)

        fig_axis[-1].set_xlabel(axis_opts.get("label", ""),  fontsize=6)

        return fig_axis, fig
