
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

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._axis = getattr(obj, '_axis', None)
        self._description = getattr(obj, '_description', None) or AttributeTree()

    def __init__(self,  value=None, *args, axis=None, description=None, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(value, (np.ndarray, int, float)):
            self.copy(value)

    @property
    def is_constant(self):
        return self.shape == ()

    def evaluate(self):
        pass

    def at(self, idx):
        return self.value[idx]

    def __getitem__(self, idx):
        res = self.value[idx]
        if isinstance(res, Profile):
            res._axis = self._axis[idx]
        elif isinstance(res, np.ndarray):
            res = Profile(res, axis=self._axis[idx])
        return res

    def __setitem__(self, *args):
        self.evaluate()
        super().__setitem__(*args)
        if hasattr(self, "interpolate"):
            del self.interpolate
        if hasattr(self, "derivative"):
            del self.derivative
        if hasattr(self, "dln"):
            del self.dln

    def iter_over(self, axis):
        if self.is_constant:
            for x in axis.flat:
                yield self.item()
        elif axis is self._axis:
            yield from self.flat
        elif isinstance(axis, np.ndarray):
            for x in axis.flat:
                yield self.eval(x)
        else:
            for x in axis:
                yield self.eval(x)

    @property
    def description(self):
        return self._description

    @property
    def axis(self):
        return self._axis

    @cached_property
    def value(self):
        if self.is_constant:
            return self.item()
        else:
            self.evaluate()
            return self.view(np.ndarray)

    @cached_property
    def interpolate(self):
        axis = self._axis.view(np.ndarray)
        data = self.value
        try:
            res = UnivariateSpline(axis, data)
        except Exception as error:
            logger.debug((error, axis, data))
            raise error
        return res

    def copy(self, other):
        if isinstance(other, ProfileExpression):
            if self._axis is other._axis:
                np.copyto(self, other.value)
            else:
                np.copyto(self, other.resample(self._axis))
        elif isinstance(other, Profile):
            np.copyto(self, other.resample(self._axis))
        elif isinstance(other, np.ndarray) and self.shape == other.shape:
            np.copyto(self, other)
        elif isinstance(other, (int, float, type(None))):
            self.fill(other)
        elif not isinstance(other, np.ndarray) and not other:
            self.fill(np.nan)
        else:
            raise ValueError(f"Can not copy object! {type(other)} {other} ")

    def resample(self, axis, *args, **kwargs):
        assert(isinstance(axis, np.ndarray))
        return self.eval(axis, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.is_constant:
            return self.value
        else:
            return self.eval(*args, **kwargs)

    def eval(self,  axis, *args, **kwargs):
        if axis is self._axis:
            return self

        ufunc = getattr(self, "_ufunc", None) or self.interpolate

        if ufunc is None:
            raise RuntimeError(f"Function is not defined")

        res = ufunc(axis, *args, **kwargs)

        if isinstance(res, Profile):
            if not hasattr(res, "_axis"):
                res._axis = axis
        elif isinstance(res, np.ndarray):
            if len(res.shape) == 0:
                res = res.item()
            elif res.size == 1:
                res = res[0]
            else:
                res = Profile(res, axis=axis, description=self.description)

        return res

        # def derivative_n(self, n, *args, **kwargs):
        #     self.evaluate()
        #     return Profile(self.interpolate.derivative(n=n)(self._axis), axis=self._axis)
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
        return Profile(np.gradient(self.value)/np.gradient(self._axis), axis=self._axis)

    @cached_property
    def dln(self, *args, **kwargs):
        r"""
            .. math:: d\ln f=\frac{df}{f}
        """
        return Profile(self.derivative.value/self.value, axis=self._axis)

    @cached_property
    def cum_integral(self):
        value = scipy.integrate.cumtrapz(self.axis, self.value, initial=0.0)
        return Profile(value, axis=self.axis)

    @cached_property
    def inv_cum_integral(self):
        value = scipy.integrate.cumtrapz(self.axis[::-1], self.value[::-1], initial=0.0)[::-1]
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
        # func = getattr(self, "_ufunc", None) or self.interpolate
        # if func is None:
        #     spl = self.interpolate
        #     if isinstance(start, np.ndarray) and isinstance(stop, np.ndarray):
        #         raise ValueError(f"Illegal arguments! start is {type(start)} ,end is {type(end)}")
        #     elif isinstance(start, np.ndarray):
        #         res = Profile(np.array([spl.integral(x, stop) for x in start]), axis=start)
        #     elif isinstance(stop, np.ndarray):
        #         res = Profile(np.array([spl.integral(start, x) for x in stop]), axis=stop)
        #     else:
        #         res = spl.integral(start, stop)
        # else:


class ProfileExpression(Profile):
    def __init__(self, ufunc, method=None, *args, func_args=None, func_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._ufunc = ufunc
        self._method = method
        self._op = getattr(ufunc, method) if method is not None else ufunc
        self._args = []

        for arg in (func_args or []):
            if isinstance(arg, Profile):
                self._args.append(arg)
            elif isinstance(arg, np.ndarray):
                assert(arg.shape == self.axis.shape)
                arg = arg.view(Profile)
                arg._axis = self.axis
                self._args.append(arg)
            else:
                self._args.append(arg)

        self._kwargs = func_kwargs or {}

    def at(self, idx):
        return self.eval(self._axis[idx])

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        # self.evaluate()
        # return super().__repr__()
        return f"<{self.__class__.__name__} ufunc={self._ufunc} method={self._method}>"

    def __call__(self, x_axis=None):
        if not isinstance(x_axis, np.ndarray):
            res = self.eval(x_axis)
        elif self.is_constant:
            return self.value
        else:
            res = np.ndarray(x_axis.shape)

            for idx, v in enumerate(self.iter_over(x_axis)):
                if isinstance(v, Profile):
                    logger.debug(v)
                    v = v.value
                try:
                    res[idx] = v
                except Exception as error:
                    logger.debug(v)
                    raise error
            # with np.nditer(res, op_flags=['readwrite']) as it:
            #     for x in it:
            #         v = next(value_it)
            #         try:
            #             x[...] = v
            #         except Exception as error:
            #             raise error

            res = res.view(Profile)
            res._axis = x_axis
            res._description = self._description

        if isinstance(res, Profile):
            pass
        elif isinstance(res, np.ndarray):
            res = Profile(res, axis=x_axis)

        return res

    def __getitem__(self, idx):

        return self.eval(self._axis[idx])

    def eval(self, x):
        if isinstance(x, (np.ndarray, collections.abc.Sequence)):
            pass
        else:
            x = [x]

        # args = [(arg(x) if callable(arg) else arg) for arg in self._args]
        args = []
        for arg in self._args:
            if callable(arg):
                args.append(arg(x))
            else:
                args.append(arg)
            if not isinstance(arg, np.ndarray) and np.isnan(arg):
                logger.error(arg)
                raise ValueError(self)

        [v for v in self.iter_over(x)]

        return self._op(*args, **self._kwargs)

    def iter_over(self, axis):
        args_it = []
        for arg in self._args:
            if isinstance(arg, Profile):
                args_it.append((arg.iter_over(axis), True))
            else:
                args_it.append((arg, False))

        for _ in np.nditer(axis):
            s_args = []
            for arg, iterable in args_it:
                if iterable:
                    s_args.append(next(arg))
                else:
                    s_args.append(arg)

            if self._method is not None:
                res = getattr(self._ufunc, self._method)(*s_args, **self._kwargs)
            else:
                res = self._ufunc(self._axis)

            if isinstance(res, ProfileExpression):
                res = res.value
                logger.debug(s_args)
                logger.debug([s for s in map(type, s_args)])
                logger.debug((self._ufunc, self._method))

            yield res
            # yield self._op(* [(next(arg) if iterable else arg) for arg, iterable in args_it], **self._kwargs)

    def __iter__(self):
        yield from self.iter_over(self._axis)

    # @property
    # def shape(self):
    #     return self._axis.shape

    @property
    def flat(self):
        yield from self.iter_over(self._axis)

    def evaluate(self):
        args = []
        in_no = []
        axis = self._axis
        for i, input_ in enumerate(self._args):
            if isinstance(input_, Profile):
                in_no.append(i)
                if axis is None:
                    axis = getattr(input_, "axis", None)

                if input_.axis is axis:
                    args.append(input_.view(np.ndarray))
                else:
                    data = input_(axis)
                    if not isinstance(data, np.ndarray):
                        args.append(data)
                    elif data.shape != ():
                        args.append(data.view(np.ndarray))
                    else:
                        args.append(data.item())

            else:
                args.append(input_)

        if super().shape != axis.shape:
            self.resize(axis.size, refcheck=False)  #
            self.reshape(axis.shape)
            self._axis = axis

        data = self.view(np.ndarray)
        for idx, v in enumerate(data.flat):
            data[idx] = v

        # outputs = out
        # out_no = []
        # if outputs:
        #     out_args = []
        #     for j, output in enumerate(outputs):
        #         if isinstance(output, Profile):
        #             out_no.append(j)
        #             out_args.append(output.view(np.ndarray))
        #         else:
        #             out_args.append(output)
        #     kwargs['out'] = tuple(out_args)
        # else:
        #     outputs = (None,) * ufunc.nout
        # info = {}
        # if in_no:
        #     info['inputs'] = in_no
        # if out_no:
        #     info['outputs'] = out_no
        # results = super(Profile, self).__array_ufunc__(ufunc, method, *args, **kwargs)
        # results = getattr(ufunc, method)(*args, **kwargs)
        # if results is NotImplemented:
        #     return NotImplemented
        # elif isinstance(results, np.ndarray) and results.shape != ():
        #     results = np.asarray(results).view(Profile)
        #     results._axis = axis
        # return results
        # if method == 'at':
        #     if isinstance(inputs[0], Profile):
        #         inputs[0].info = info
        #     return
        # if ufunc.nout == 1:
        #     results = (results,)
        # res = []
        # for result, output in zip(results, outputs):
        #     if output is not None:
        #         res.append(output)
        #     elif isinstance(result, np.ndarray) and result.shape != ():
        #         p = np.asarray(result).view(Profile)
        #         p._axis = axis
        #         res.append(p)
        #     else:
        #         res.append(result)
        # return res[0] if len(res) == 1 else res


def _profile_new_(cls, *args, axis=None, description=None,  **kwargs):

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

    shape = axis.shape if axis is not None else ()

    if len(args) == 0 or args[0] is None or isinstance(args[0],  (np.ndarray, int, float)):
        pass
    else:
        cls = ProfileExpression

    obj = super(Profile, cls).__new__(cls, shape)
    obj._axis = axis
    obj._description = AttributeTree(description)
    return obj


Profile.__new__ = _profile_new_
Profile.__array_ufunc__ = lambda s, ufunc, method, *args, axis=None, **kwargs: ProfileExpression(
    ufunc, method,
    func_args=args, func_kwargs=kwargs,
    axis=axis if axis is not None else s._axis)


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
        v = self.__getitem__(key)
        if v is None:
            self.__as_object__()[key] = self._create(value, name=key)
        elif isinstance(v, Profile):
            if isinstance(value, (np.ndarray, int, float)):
                v.copy(value)
            elif callable(value):
                ufunc = np.vectorize(value)
                v.copy(ufunc(self._axis))
            else:
                raise ValueError(value)
        else:
            raise KeyError(f"Can not assign value to {key}: {type(v)}!")

    @lru_cache
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
