
import collections
import copy
import inspect
import numbers
import types
from functools import cached_property, lru_cache

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.constants
import scipy.integrate
import scipy.misc
from scipy.interpolate import RectBivariateSpline, UnivariateSpline, interp1d
from spdm.util.AttributeTree import AttributeTree
from spdm.util.LazyProxy import LazyProxy
from spdm.util.logger import logger


class Profile(np.ndarray):

    @staticmethod
    def __new__(cls, *args, axis=None, description=None,  **kwargs):
        if isinstance(axis, Profile):
            axis = axis.value
        elif axis is None:
            pass
        elif isinstance(axis, np.ndarray):
            pass
        elif isinstance(axis, int):
            axis = np.linspace(0.0, 1.0, axis)
        elif isinstance(axis, collections.abc.Sequence):
            axis = np.array(axis)

        else:
            raise ValueError(f"Illegal axis! {axis}")

        if cls is ProfileExpression:
            for arg in kwargs.get("func_args", []):
                if isinstance(arg, Profile) and axis is None:
                    axis = arg.axis

        shape = axis.shape if axis is not None else ()

        if len(args) == 0:
            pass
        elif isinstance(args[0],  (np.ndarray, int, float)) or not args[0]:
            pass
        elif cls is ProfileExpression:
            pass

        elif callable(args[0]) or isinstance(args[0], np.ufunc):
            cls = ProfileFunction
        # else:
        #     raise TypeError(f"illegal value {type(args[0])} {args[0]}")

        obj = super(Profile, cls).__new__(cls, shape)
        obj._axis = axis
        obj._description = AttributeTree(description)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._axis = getattr(obj, '_axis', None)
        self._description = getattr(obj, '_description', None) or AttributeTree()

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        if any([isinstance(arg, ProfileFunction) for arg in inputs]):
            def op(*args, _holder=self,  _ufunc=ufunc, _method=method, **kwargs):
                res = super(Profile, _holder).__array_ufunc__(_ufunc, _method, *args, out=out, **kwargs)
                if res is NotImplemented:
                    res = getattr(_ufunc, _method)(*args,  **kwargs)
                    if res is NotImplemented:
                        raise RuntimeError((_holder, _ufunc, _method))
                return res

            return ProfileExpression(op, func_args=inputs, func_kwargs=kwargs)

        x_axis = self._axis

        args = []
        for arg in inputs:
            if isinstance(arg,  Profile):
                data = arg(x_axis)
            else:
                data = arg
            if isinstance(data, Profile):
                args.append(data.view(np.ndarray))
            else:
                args.append(data)

        res = super(Profile, self).__array_ufunc__(ufunc, method, *args, **kwargs)

        if isinstance(res, np.ndarray) and res.shape != () and (any(np.isnan(res)) or any(np.isinf(res))):
            raise ValueError(res)

        if isinstance(res, np.ndarray) and not isinstance(res, Profile):
            res = res.view(Profile)
            res._axis = x_axis
            # return super(Profile, self).__array_ufunc__(ufunc, method, *args, out=out, **kwargs)
        return res

    def __init__(self,  value=None, *args, axis=None, description=None, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(value, Profile):
            value = value(self._axis).view(np.ndarray)

        if isinstance(value, (np.ndarray, int, float)):
            self.copy(value)

    def __repr__(self):
        desc = getattr(self, "_description", None)
        if desc is not None:
            return f"<{self.__class__.__name__} name='{ desc.name}'>"
        else:
            return super().__repr__()

    @property
    def is_constant(self):
        return self.shape == () and self.__class__ is not ProfileFunction

    @property
    def value(self):
        if self.is_constant:
            return self.item()
        else:
            return self.view(np.ndarray)

    def __setitem__(self, *args):
        super().__setitem__(*args)
        if hasattr(self, "as_function"):
            del self.as_function
        if hasattr(self, "derivative"):
            del self.derivative
        if hasattr(self, "dln"):
            del self.dln

    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        if isinstance(res, np.ndarray):
            res = res.view(np.ndarray)
        return res

    @property
    def description(self):
        return self._description

    @property
    def axis(self):
        return self._axis

    @cached_property
    def as_function(self):
        axis = self._axis.view(np.ndarray)
        data = self.value
        try:
            res = interp1d(axis, data, kind=self.metadata.interpolator or 'linear')
        except Exception as error:
            logger.debug((error, axis, data))
            raise error
        return res

    def copy(self, other):
        if isinstance(other, Profile):
            if self._axis is other._axis:
                np.copyto(self, other.value)
            else:
                np.copyto(self, other(self._axis))
        elif not isinstance(other, np.ndarray):
            self.fill(other)
        elif self.shape == other.shape:
            np.copyto(self, other)
        else:
            raise ValueError(f"Can not copy object! {type(other)} {other} ")

    def __call__(self, x_axis=None, *args, **kwargs):
        if x_axis is self._axis or x_axis is None:
            return self
        res = self.as_function(x_axis)
        if isinstance(res, Profile):
            if not hasattr(res, "_axis"):
                res._axis = x_axis
        elif isinstance(res, np.ndarray):
            if len(res.shape) == 0:
                res = res.item()
            elif res.size == 1:
                res = res[0]
            else:
                res = res.view(Profile)
                res._axis = x_axis
                res._description = self.metadata

        return res

        # def derivative_n(self, n, *args, **kwargs):
        #     self.evaluate()
        #     return Profile(self.as_function.derivative(n=n)(self._axis), axis=self._axis)
        # func = getattr(self, "_ufunc", None)
        # if func is None:
        # elif callable(func) or isinstance(func, types.BuiltinFunctionType):
        #     v0 = scipy.misc.derivative(func, self._axis[0], dx=self._axis[1]-self._axis[0], n=n, **kwargs)
        #     vn = scipy.misc.derivative(func, self._axis[-1], dx=self._axis[-1]-self._axis[-2], n=n, **kwargs)
        #     v = [scipy.misc.derivative(func, x, dx=0.5*(self._axis[i+1]-self._axis[i-1]),
        #                                n=n, *args, **kwargs) for i, x in enumerate(self._axis[1:-1])]
        #     return Profile(np.array([v0]+v+[vn]), axis=self._axis)

    @cached_property
    def derivative(self):
        # value = UnivariateSpline(self._axis, self.value).derivative()(self._axis)
        # return Profile(value[:], axis=self._axis)
        return Profile(np.gradient(self[:])/np.gradient(self._axis[:]), axis=self._axis)

    # @cached_property
    # def dln(self, *args, **kwargs):
    #     r"""
    #         .. math:: d\ln f=\frac{df}{f}
    #     """
    #     data = np.ndarray(self._axis.shape)
    #     data[1:] = self.derivative.value[1:]/self.value[1:]
    #     data[0] = 2*data[1]-data[2]
    #     if any(np.isnan(data)) or any(self.value == 0):
    #         logger.error(self.value)
    #         raise ValueError(data)
    #     return Profile(data, axis=self._axis)

    @cached_property
    def integral(self):
        return Profile(scipy.integrate.cumtrapz(self.value, self.axis, initial=0.0), axis=self.axis)

    @cached_property
    def inv_integral(self):
        value = scipy.integrate.cumtrapz(self.value[::-1], self.axis[::-1], initial=0.0)[::-1]
        return Profile(value, axis=self.axis)

        # if isinstance(start, np.ndarray) and isinstance(stop, np.ndarray):
        #     raise ValueError(f"Illegal arguments! start is {type(start)} ,end is {type(stop)}")
        # elif isinstance(start, np.ndarray):
        #     value =self.

        #     res = Profile(np.array([scipy.integrate.quad(func, x, stop)[0] for x in start]), axis=start)
        # elif isinstance(stop, np.ndarray):
        #     res = Profile(np.array([scipy.integrate.quad(func, start, x)[0] for x in stop]), axis=stop)
        # else:
        #     res = scipy.integrate.quad(func, start, stop)
        # return res
        # self.evaluate()
        # if start is None:
        #     start = self._axis
        # if stop is None:
        #     stop = self._axis
        # func = getattr(self, "_ufunc", None) or self.as_function
        # if func is None:
        #     spl = self.as_function
        #     if isinstance(start, np.ndarray) and isinstance(stop, np.ndarray):
        #         raise ValueError(f"Illegal arguments! start is {type(start)} ,end is {type(end)}")
        #     elif isinstance(start, np.ndarray):
        #         res = Profile(np.array([spl.integral(x, stop) for x in start]), axis=start)
        #     elif isinstance(stop, np.ndarray):
        #         res = Profile(np.array([spl.integral(start, x) for x in stop]), axis=stop)
        #     else:
        #         res = spl.integral(start, stop)
        # else:


class ProfileFunction(Profile):

    def __init__(self, ufunc,   *args,  **kwargs):
        super().__init__(*args, **kwargs)

        if not callable(ufunc):
            raise TypeError(type(ufunc))
        elif not isinstance(ufunc, np.ufunc):
            ufunc = np.vectorize(ufunc)
        self._ufunc = ufunc

    def __call__(self, x=None):
        if x is None:
            x = self._axis
        if x is None:
            raise ValueError(f" x is None !")

        res = self._ufunc(x)
        if isinstance(res, np.ndarray) and not isinstance(res, Profile):
            res = res.view(Profile)
            res._axis = x
        return res

    def __getitem__(self, idx):
        return self._ufunc(self._axis[idx])

    @property
    def value(self):
        if self.shape == ():
            try:
                self.resize(self._axis.size, refcheck=True)
                self.reshape(self._axis.shape)
            except Exception:
                res = self._ufunc(self._axis)
            else:
                np.copyto(self, self._ufunc(self._axis))
                res = self.view(np.ndarray)
        else:
            res = self.view(np.ndarray)
        return res

    @cached_property
    def derivative(self):
        return Profile(np.gradient(self.value)/np.gradient(self._axis), axis=self._axis)

    @cached_property
    def dln(self, *args, **kwargs):
        r"""
            .. math:: d\ln f=\frac{df}{f}
        """
        return Profile(self.derivative.value/self.value, axis=self._axis)

    @cached_property
    def integral(self):
        data = scipy.integrate.cumtrapz(self[:], self.axis[:], initial=0.0)
        return Profile(data, axis=self.axis)

    @cached_property
    def inv_integral(self):
        value = scipy.integrate.cumtrapz(self[::-1], self.axis[::-1], initial=0.0)[::-1]
        return Profile(value, axis=self.axis)


class ProfileExpression(Profile):

    def __init__(self,    func,   *args, func_args=None, func_kwargs=None,  **kwargs):
        self._func = func
        self._args = func_args or []
        self._kwargs = func_kwargs or {}
        super().__init__(*args, **kwargs)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"<{self.__class__.__name__} ufunc={self._func}  >"

    def __call__(self, x_axis=None):
        if x_axis is None:
            x_axis = self._axis
        args = []
        for arg in self._args:
            if isinstance(arg,  Profile):
                data = arg(x_axis)
            else:
                data = arg
            if isinstance(data, Profile):
                args.append(data.view(np.ndarray))
            else:
                args.append(data)

        res = self._func(*args, **self._kwargs)

        if isinstance(res, np.ndarray) and res.shape != () and (any(np.isnan(res)) or any(np.isinf(res))):
            raise ValueError(res)

        if isinstance(res, np.ndarray) and not isinstance(res, Profile):
            res = res.view(Profile)
            res._axis = x_axis

        return res

    def __getitem__(self, idx):
        args = []
        if not hasattr(self, "_args"):
            return self.view(np.ndarray)[idx]

        for arg in self._args:
            if not isinstance(arg,  np.ndarray):
                args.append(arg)
            elif not isinstance(arg, Profile):
                args.append(arg[idx])
            elif arg.axis is self.axis:
                args.append(arg[idx])
            else:
                data = arg(self._axis[idx])
                if isinstance(data, np.ndarray):
                    args.append(data.view(np.ndarray))
                else:
                    args.append(data)

        return self._func(*args, **self._kwargs)

    def __setitem__(self, idx, value):
        if self.shape != self._axis.shape:
            self.evaluate()
        self.view(np.ndarray)[idx] = value

    def evaluate(self):
        if self.shape != self._axis.shape:
            self.resize(self._axis.size, refcheck=False)
            self.reshape(self._axis.shape)
        np.copyto(self, self[:])

    @ property
    def value(self):
        return self[:]


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

        self.__dict__["_axis"] = axis

    def _create(self, d=None, name=None, **kwargs):
        if isinstance(d, Profile) and not hasattr(d, "_axis"):
            d._axis = self._axis
        else:
            d = Profile(d, axis=self._axis, description={"name": name, **kwargs})
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
        if isinstance(value, Profile):
            self.__as_object__()[key] = value(self._axis)
        else:
            self.__as_object__()[key] = Profile(value, axis=self._axis,  description={"name": key})
        # v = self.__getitem__(key)
        # if v is None:
        # elif isinstance(v, Profile):
        #     if isinstance(value, (np.ndarray, int, float)):
        #         v.copy(value)
        #     elif callable(value):
        #         ufunc = np.vectorize(value)
        #         v.copy(ufunc(self._axis))
        #     else:
        #         raise ValueError(value)
        # else:
        #     raise KeyError(f"Can not assign value to {key}: {type(v)}!")

    @ lru_cache
    def cache(self, key):
        res = self._cache[key.split(".")]
        if isinstance(res, LazyProxy):
            res = res()
        return res

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
