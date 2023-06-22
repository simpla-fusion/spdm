import collections.abc
import typing
from dataclasses import dataclass

import numpy as np
import numpy.typing as np_tp

PrimaryType = int | float | bool | complex | str | bytes

ArrayLike = np_tp.ArrayLike | None

ScalarType = bool | int | float | complex | np.float64 | np.complex64 | np.complex128

ArrayType = np_tp.NDArray[np.floating | np.complexfloating]

NumericType = ScalarType | ArrayType | None

boolean_type = (bool,)

integral_type = (int, np.integer,)

real_type = (float, np.floating,)


bitwise_and_reduce = np.bitwise_and.reduce

complex_type = (complex, np.complexfloating,)

scalar_type = (*boolean_type, *integral_type, *real_type, *complex_type,)

array_type = np.ndarray



# def is_arraylike(d: typing.Any) -> bool:
#     return is_scalarlike(d) or isinstance(d, (collections.abc.Sequence, np.ndarray)) or hasattr(d.__class__, "__array__")


np_squeeze = np.squeeze

numeric_type = (*scalar_type, array_type)

primary_type = (str,  *numeric_type)

_T = typing.TypeVar("_T")

HTContainer = _T | typing.Sequence[_T]
"""Hierarchical Container Type  """


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
