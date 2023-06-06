import collections.abc
import typing
from dataclasses import dataclass

import numpy as np
import numpy.typing as np_tp

PrimaryType = int | float | bool | complex | str | bytes

ArrayLike = np_tp.ArrayLike

ScalarType = float | complex | np.float64 | np.complex64 | np.complex128

ArrayType = np_tp.NDArray[np.floating | np.complexfloating]

NumericType = ScalarType | ArrayType

boolean_type = (bool,)

integral_type = (int, np.integer,)

real_type = (float, np.floating,)

complex_type = (complex, np.complexfloating,)

scalar_type = (*boolean_type, *integral_type, *real_type, *complex_type,)

array_type = np.ndarray

numeric_type = (*scalar_type, array_type)

primary_type = (str,  *numeric_type)


def is_scalar(v: typing.Any) -> bool:
    return isinstance(v, scalar_type) or (isinstance(v, array_type) and len(v.shape) == 0)


def is_vector(v: typing.Any) -> bool:
    return isinstance(v, collections.abc.Sequence) and all(is_scalar(d) for d in v)


def is_array(v: typing.Any) -> bool:
    return (isinstance(v, array_type) and len(v.shape) > 0)


def is_numeric(v: typing.Any) -> bool:
    return isinstance(v, numeric_type)


@dataclass
class Vector2:
    x: float
    y: float


@dataclass
class Vector3:
    x: float
    y: float
    z: float


@dataclass
class Vector4:
    x: float
    y: float
    z: float
    t: float


nTupleType = typing.Tuple[ScalarType, ...]
_T = typing.TypeVar("_T")


class Vector(typing.Tuple[_T]):
    """ Vector 矢量
    --------------
    用于描述一个矢量（流形上的切矢量。。。），
    _T: typing.Type 矢量元素的类型, 可以是实数，也可以是复数
    """
    pass


def is_complex(d: typing.Any) -> bool:
    return np.iscomplexobj(d)


def is_real(d: typing.Any) -> bool:
    return not np.iscomplexobj(d)


def is_scalarlike(d: typing.Any) -> bool:
    return isinstance(d, (int, float, complex, np.floating, np.complexfloating)) or hasattr(d.__class__, "__float__")


def as_scalar(d: typing.Any) -> ScalarType:
    return complex(d) if is_complex(d) else float(d)


def is_arraylike(d: typing.Any) -> bool:
    return is_scalarlike(d) or isinstance(d, (collections.abc.Sequence, np.ndarray)) or hasattr(d.__class__, "__array__")


def as_array(d: typing.Any, **kwargs) -> np_tp.NDArray:
    if hasattr(d.__class__, '__entry__'):
        d = d.__entry__().__value__()
    return np.asarray(d, **kwargs)
