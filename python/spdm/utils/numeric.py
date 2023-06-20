import collections.abc
import typing

import numpy as np

from .typing import ScalarType, array_type, numeric_type

float_nan = np.nan
is_scalar = np.isscalar
is_close = np.isclose


def is_vector(v: typing.Any) -> bool:
    return isinstance(v, collections.abc.Sequence) and all(is_scalar(d) for d in v)


def is_array(v: typing.Any) -> bool:
    return (isinstance(v, array_type) and len(v.shape) > 0)


def is_numeric(v: typing.Any) -> bool:
    return isinstance(v, numeric_type)


def is_complex(d: typing.Any) -> bool:
    return np.iscomplexobj(d)


def is_real(d: typing.Any) -> bool:
    return not np.iscomplexobj(d)


def is_scalarlike(d: typing.Any) -> bool:
    return isinstance(d, (int, float, complex, np.floating, np.complexfloating)) or hasattr(d.__class__, "__float__")


def as_scalar(d: typing.Any) -> ScalarType:
    return complex(d) if is_complex(d) else float(d)


def as_array(d: typing.Any, **kwargs) -> array_type:
    if hasattr(d.__class__, '__value__'):
        d = d.__value__
    return np.asarray(d, **kwargs)


meshgrid = np.meshgrid

bitwise_and = np.bitwise_and


squeeze = np.squeeze

float_nan = np.nan
