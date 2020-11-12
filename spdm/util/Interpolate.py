from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from spdm.util.sp_export import sp_find_module
from spdm.util.logger import logger


class Interpolate(object):

    @classmethod
    def __new__(cls, *args, **kwargs):
        if hasattr(Interpolate, "_create_interpolate"):
            return Interpolate._create_interpolate(cls, *args, **kwargs)
        else:
            return object.__new__(cls)

    def __init__(self,   *args,   **kwargs):
        pass


class Interpolate1D(Interpolate):
    def __init__(self, x, y, *args, k=3, copy=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._fun = UnivariateSpline(x,  y, *args,   copy=copy, **kwargs)

    def __call__(self, x=None, *args, **kwargs):
        if x is None:
            return self._fun
        else:
            return self._fun(x)

    def derivate(self, *args, n=1, **kwargs):
        return self._fun.derivative(n)


class Interpolate2D(Interpolate):
    def __init__(self,  x, y, z, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self._fun = RectBivariateSpline(x,  y,  z, *args,  **kwargs)

    def __call__(self, x=None, y=None, grid=False, **kwargs):
        if x is None or y is None:
            return self._fun
        else:
            return self._fun(x, y, grid=grid, **kwargs)

    def derivate(self, x, y, * args, dx=1, dy=1, **kwargs):
        return self._fun(x, y, * args, dx=dx, dy=dy, **kwargs)

    def dx(self, x, y, dx=1, grid=False, **kwargs):
        return self._fun(x, y,   dx=dx, grid=grid, **kwargs)

    def dy(self, x, y, dy=1, grid=False, **kwargs):
        return self._fun(x, y,   dy=dy, grid=grid, **kwargs)


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
    if isinstance(y, Interpolate1D):
        pass
    elif x is not None:
        y = Interpolate1D(x, y)
    else:
        raise TypeError(f"y={type(y)} x={type(x)}")

    return y.derivate(*args, **kwargs)


def integral(y, x, *args, **kwargs):
    return NotImplemented
