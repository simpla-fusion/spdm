from scipy import interpolate as interp
from spdm.util.sp_export import sp_find_module
from spdm.util.logger import logger


class Interpolate(object):

    @classmethod
    def __new__(cls, *args, **kwargs):
        if hasattr(Interpolate, "_create_interpolate"):
            return Interpolate._create_interpolate(cls, *args, **kwargs)
        else:
            return object.__new__(cls)

    def __init__(self, y,  x, *args, copy=False, k=3, **kwargs):
        self._x = x
        self._y = y


class Interpolate1D(Interpolate):
    def __init__(self, y,  x, *args, copy=False, k=3, **kwargs):
        super().__init__(y, x, *args, **kwargs)

    def fun(self, *args, copy=False, k=3, **kwargs):
        return interp.UnivariateSpline(self._x, self._y, *args, k=k, **kwargs)

    def __call__(self, x=None, *args, **kwargs):
        if x is None:
            return self._y, self._x
        else:
            return self.fun(*args, **kwargs)(x)

    def derivate(self, n=1, *args, **kwargs):
        return self.fun(*args, **kwargs).derivative(n)


class Interpolate2D(Interpolate):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class InterpolateND(Interpolate):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


def _create_interpolate(cls, y,  x=None, *args, backend=None, ** kwargs):
    if cls is not Interpolate:
        return super(Interpolate, cls).__new__(cls)

    if backend is not None:
        plugin_name = f"{__package__}.plugins.interpolate.Plugin{backend}"

        n_cls = sp_find_module(plugin_name, fragment=f"Interpolate{backend}")

        if n_cls is None:
            raise ModuleNotFoundError(f"Can not find plugin {plugin_name}#Interpolate{backend}")
    elif len(x.shape) == 1:
        n_cls = Interpolate1D
    elif len(x.shape) == 2:
        n_cls = Interpolate2D
    else:
        n_cls = InterpolateND

    return object.__new__(n_cls)


Interpolate._create_interpolate = _create_interpolate


def interpolate(y, x, *args, **kwargs):
    if isinstance(y, Interpolate1D):
        return y.reset(x, *args, **kwargs)
    else:
        return Interpolate(y, x, *args, **kwargs)


def derivate(y, x=None, *args, **kwargs):
    if not isinstance(y, Interpolate1D):
        y = Interpolate1D(x, y)
        return y.derivate(*args, **kwargs)(x)
    elif x is not None:
        return y.derivate(*args, **kwargs)(x)
    else:
        return y.derivate(*args, **kwargs)()


def integral(y, x, *args, **kwargs):
    return NotImplemented
