from __future__ import annotations

import collections.abc
import inspect
import typing
from copy import copy
from enum import Enum
from functools import cached_property, lru_cache

import numpy as np
from scipy.interpolate import (InterpolatedUnivariateSpline, PPoly,
                               RectBivariateSpline)
from spdm.utils.typing import ArrayType, NumericType

from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix
from ..utils.tags import _not_found_
from ..utils.typing import (ArrayType, NumericType, ScalarType, as_array,
                            as_scalar)

_T = typing.TypeVar("_T")


class Function(typing.Generic[_T]):
    """
    Function 函数
    --------------
    用于描述一个函数（流形上的映射。。。），
    - 可以是一个数值函数，也可以是一个插值函数
    - 可以是一个标量函数，也可以是一个矢量函数
    - 建立在一维或者多维网格上

    _Mesh: Mesh 以网格的形式描述函数所在流形，
        - Mesh.points 网格点坐标

    _data: np.ndarray | typing.Callable[..., NumericType]
        - 网格点上数值 DoF
        - 描述函数的数值或者插值函数

    TODO:
        - Function[ScalarTypeValue,ScalarTypeMesh] 两个泛型参数，分别描述数值和网格的类型
    """

    def __init__(self, d=None, *args, mesh=None, **kwargs):
        """
        初始化Function 函数
        --------------
        d: np.ndarray | typing.Callable[..., NumericType]
            - 网格点上数值 DoF
            - 描述函数的数值或者插值函数

        """

        self._data = d  # 网格点上的数值 DoF

        mesh_desc, self._metadata = group_dict_by_prefix(kwargs, "mesh_")

        if isinstance(mesh, collections.abc.Mapping):
            mesh_desc.update(mesh)
            mesh = None
        elif isinstance(mesh, Enum):
            mesh_desc.update({"type": mesh.name})
            mesh = None
        elif isinstance(mesh, str):
            mesh_desc.update({"type": mesh})
            mesh = None

        if mesh is not None and mesh is not _not_found_:
            self._mesh = mesh
            if len(mesh_desc) > 0:
                logger.warning(f"self._mesh is specified, ignore mesh_desc={mesh_desc}")
        elif len(mesh_desc) == 0:
            self._mesh = args if len(args) != 1 else args[0]
        else:
            try:
                from ..mesh.Mesh import Mesh
                self._mesh = Mesh(*args, **mesh_desc)
            except ModuleNotFoundError:
                raise RuntimeError(f"Can not import Mesh from spdm.mesh.Mesh!")
            except:
                raise RuntimeError(f"Can not create mesh from mesh_desc={mesh_desc}")

        self._ppoly_cache = {}

    @property
    def metadata(self):
        return self._metadata

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}  mesh_type=\"{self._mesh.__class__.__name__}\" data_type=\"{self.__type_hint__.__name__}\" />"

    def __serialize__(self) -> typing.Mapping:
        raise NotImplementedError(f"")

    @classmethod
    def __deserialize__(cls, data: typing.Mapping) -> Function:
        raise NotImplementedError(f"")

    def __copy__(self) -> Function:
        """复制 Function """
        other = object.__new__(self.__class__)
        other._mesh = self._mesh
        other._data = copy(self._data)
        return other

    @property
    def mesh(self): return self._mesh

    @property
    def domain(self): return getattr(self.mesh, "geometry", self.bbox)
    """ 返回函数的定义域，即函数参数的取值范围。
        - 如果mesh有geometry属性，则返回这个属性    
        - 否则返回 bbox
    """

    @property
    def bbox(self) -> typing.Tuple[ArrayType, ArrayType]: return getattr(self._mesh, "bbox", None)
    """ bound box 返回包裹函数参数的取值范围的最小多维度超矩形（hyperrectangle） """

    def __array__(self) -> ArrayType:
        """ 重载 numpy 的 __array__ 运算符"""

        if isinstance(self._data, np.ndarray):
            pass
        elif hasattr(self._data, "__entry__"):
            self._data = np.asarray(self._data.__entry__().__value__(), dtype=float)
        else:
            raise RuntimeError(f"Can not convert {self._data} to np.ndarray!")

        return self._data

    def __getitem__(self, *args) -> NumericType:
        return self.__array__().__getitem__(*args)

    def __setitem__(self, *args) -> None:
        return self.__array__().__setitem__(*args)

    # fmt:off
    def __array_ufunc__(self, ufunc, method, *args, **kwargs) -> Expression: return Expression(*args, ufunc=ufunc, method=method, **kwargs)
    """ 重载 numpy 的 ufunc 运算符"""
    # fmt:on

    @cached_property
    def __type_hint__(self) -> typing.Type:
        """ 获取函数的类型
        """
        orig_class = getattr(self, "__orig_class__", None)
        if orig_class is not None:
            return typing.get_args(orig_class)[0]
        elif isinstance(self._data, np.ndarray):
            return self._data.dtype.type
        elif callable(self._data):
            return typing.get_type_hints(self._data).get("return", None)
        else:
            return None

    def __ppoly__(self, *dx: int, **kwargs) -> typing.Callable[..., NumericType]:
        """ 返回 PPoly 对象 """
        fun = self._ppoly_cache.get(dx, None)
        if fun is not None:
            return fun

        if len(dx) == 0:
            if callable(self._data):
                fun = self._data
            elif hasattr(self._mesh, "interpolator"):
                fun = self._mesh.interpolator(self.__array__())
            elif isinstance(self._mesh, np.ndarray):
                fun = InterpolatedUnivariateSpline(self._mesh, self._data,  **kwargs)
            elif isinstance(self._mesh, tuple) and len(self._mesh) == 2:
                fun = RectBivariateSpline(*self._mesh, self._data,  **kwargs)
            else:
                raise NotImplementedError(f"NOT IMPLEMENTED interpolate ndim>2")
        else:
            ppoly = self.__ppoly__()

            if all(d < 0 for d in dx):
                if hasattr(ppoly.__class__, 'antiderivative'):
                    fun = self.__ppoly__().antiderivative(*dx)
                elif hasattr(self._mesh, "antiderivative"):
                    fun = self._mesh.antiderivative(self.__array__(), *dx)
            elif all(d >= 0 for d in dx):
                if hasattr(ppoly.__class__, 'partial_derivative'):
                    fun = self.__ppoly__().partial_derivative(*dx)
                elif hasattr(self._mesh, "partial_derivative"):
                    fun = self._mesh.partial_derivative(self.__array__(), *dx)

        if fun is None:
            raise RuntimeError(f"Can not convert Function to PPoly! mesh_type={type(self._mesh)}")

        self._ppoly_cache[dx] = fun

        return fun

    def __call__(self, *args, ** kwargs) -> NumericType: return self.__ppoly__()(*args, grid=False, ** kwargs)

    def partial_derivative(self, *dx) -> Function: return Function(self.__ppoly__(*dx), self._mesh)

    def pd(self, *dx) -> Function: return self.partial_derivative(*dx)

    def antiderivative(self, *dx) -> Function: return Function(self.__ppoly__(*dx), self._mesh)

    def integral(self, *dx) -> Function: return self.antiderivative(*dx)

    def dln(self, *dx) -> Function:
        # v = self._interpolator(self._mesh)
        # x = (self._mesh[:-1]+self._mesh[1:])*0.5
        # return Function(x, (v[1:]-v[:-1]) / (v[1:]+v[:-1]) / (self._mesh[1:]-self._mesh[:-1])*2.0)
        return self.pd(*dx) / self

    def integrate(self, *args, **kwargs) -> ScalarType:
        return as_scalar(self._mesh.integrate(self._data, *args, **kwargs))



    # fmt: off
    def __neg__      (self                           ) : return Expression((self,)   ,self._mesh, ufunc=np.negative     )
    def __add__      (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.add          )
    def __sub__      (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.subtract     )
    def __mul__      (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.multiply     )
    def __matmul__   (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.matmul       )
    def __truediv__  (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.true_divide  )
    def __pow__      (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.power        )
    def __eq__       (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.equal        )
    def __ne__       (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.not_equal    )
    def __lt__       (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.less         )
    def __le__       (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.less_equal   )
    def __gt__       (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.greater_equal)
    def __ge__       (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.greater_equal)
    def __radd__     (self, o: NumericType | Function) : return Expression((o, self) ,self._mesh, ufunc=np.add          )
    def __rsub__     (self, o: NumericType | Function) : return Expression((o, self) ,self._mesh, ufunc=np.subtract     )
    def __rmul__     (self, o: NumericType | Function) : return Expression((o, self) ,self._mesh, ufunc=np.multiply     )
    def __rmatmul__  (self, o: NumericType | Function) : return Expression((o, self) ,self._mesh, ufunc=np.matmul       )
    def __rtruediv__ (self, o: NumericType | Function) : return Expression((o, self) ,self._mesh, ufunc=np.divide       )
    def __rpow__     (self, o: NumericType | Function) : return Expression((o, self) ,self._mesh, ufunc=np.power        )
    def __abs__      (self                           ) : return Expression((self,)   ,self._mesh, ufunc=np.abs          )
    def __pos__      (self                           ) : return Expression((self,)   ,self._mesh, ufunc=np.positive     )
    def __invert__   (self                           ) : return Expression((self,)   ,self._mesh, ufunc=np.invert       )
    def __and__      (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.bitwise_and  )
    def __or__       (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.bitwise_or   )
    def __xor__      (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.bitwise_xor  )
    def __rand__     (self, o: NumericType | Function) : return Expression((o, self) ,self._mesh, ufunc=np.bitwise_and  )
    def __ror__      (self, o: NumericType | Function) : return Expression((o, self) ,self._mesh, ufunc=np.bitwise_or   )
    def __rxor__     (self, o: NumericType | Function) : return Expression((o, self) ,self._mesh, ufunc=np.bitwise_xor  )
    def __rshift__   (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.right_shift  )
    def __lshift__   (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.left_shift   )
    def __rrshift__  (self, o: NumericType | Function) : return Expression((o, self) ,self._mesh, ufunc=np.right_shift  )
    def __rlshift__  (self, o: NumericType | Function) : return Expression((o, self) ,self._mesh, ufunc=np.left_shift   )
    def __mod__      (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.mod          )
    def __rmod__     (self, o: NumericType | Function) : return Expression((o, self) ,self._mesh, ufunc=np.mod          )
    def __floordiv__ (self, o: NumericType | Function) : return Expression((self, o) ,self._mesh, ufunc=np.floor_divide )
    def __rfloordiv__(self, o: NumericType | Function) : return Expression((o, self) ,self._mesh, ufunc=np.floor_divide )
    def __trunc__    (self                           ) : return Expression((self,)   ,self._mesh, ufunc=np.trunc        )
    def __round__    (self, n=None                   ) : return Expression((self, n) ,self._mesh, ufunc=np.round        )
    def __floor__    (self                           ) : return Expression((self,)   ,self._mesh, ufunc=np.floor        )
    def __ceil__     (self                           ) : return Expression((self,)   ,self._mesh, ufunc=np.ceil         )
    # fmt: on


class Expression(Function):
    def __init__(self, *args,  ufunc: typing.Callable | None = None, method: str | None = None, **kwargs) -> None:
        if ufunc is None and len(args) > 0 and callable(args[0]):
            ufunc = args[0]
            args = args[1:]
        super().__init__(*args, **kwargs)
        self._ufunc = ufunc
        self._method = method

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}   op=\"{self._ufunc.__name__}\" />"

    def __array__(self) -> ArrayType: raise RuntimeError("Expression cannot be converted to array without arguments!")

    def __getitem__(self, *args) -> NumericType: raise RuntimeError("Expression cannot be indexed!")

    def __setitem__(self, *args) -> None: raise RuntimeError("Expression cannot be indexed!")

    def __call__(self,  *args: NumericType, **kwargs) -> ArrayType:
        try:
            dtype = self.__type_hint__
        except TypeError:
            dtype = float

        if not inspect.isclass(dtype):
            dtype = float

        if isinstance(self._data, collections.abc.Sequence):
            value = [(d(*args, **kwargs) if callable(d) else d) for d in self._data]
        elif callable(self._data):
            value = [self._data(*args, **kwargs)]
        else:
            value = [self._data]

        if self._method is not None:
            ufunc = getattr(self._ufunc, self._method, None)
            if ufunc is None:
                raise AttributeError(f"{self._ufunc.__class__.__name__} has not method {self._method}!")
            return ufunc(self, *value)
        elif callable(self._ufunc):
            return self._ufunc(*value)  # type: ignore
        else:
            raise ValueError(f"ufunc is not callable ufunc={self._ufunc} method={self._method}")


def function_like(y: NumericType, *args: NumericType, **kwargs) -> Function:
    if len(args) == 0 and isinstance(y, Function):
        return y
    else:
        return Function(y, *args, **kwargs)


_0 = Expression(lambda *args:   args[0])
_1 = Expression(lambda *args:   args[1])
_2 = Expression(lambda *args:   args[2])
_3 = Expression(lambda *args:   args[3])
_4 = Expression(lambda *args:   args[4])
_5 = Expression(lambda *args:   args[5])
_6 = Expression(lambda *args:   args[6])
_7 = Expression(lambda *args:   args[7])
_8 = Expression(lambda *args:   args[8])
_9 = Expression(lambda *args:   args[9])


class PiecewiseFunction(Function):
    """ PiecewiseFunction
        ----------------
        A piecewise function. 一维或多维，分段函数
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if not all(isinstance(arg, collections.abc.Sequence) for arg in args):
            raise TypeError(f"PiecewiseFunction only support sequence of (cond, fun), {args}")
        self._fun_list = zip(*args)

    def __call__(self, x) -> NumericType:
        if isinstance(x, float):
            res = [fun(x) for fun, cond in self._fun_list if cond(x)]
            return res[0]
        elif isinstance(x, np.ndarray) and x.ndim == 1:
            res = np.hstack([fun(x[cond(x)]) for fun, cond in self._fun_list])
            if len(res) != len(x):
                raise RuntimeError(f"PiecewiseFunction result length not equal to input length, {len(res)}!={len(x)}")
            return res
        else:
            raise TypeError(f"PiecewiseFunction only support single float or  1D array, {x}")

    def derivative(self, *args, **kwargs) -> NumericType | Function:
        return super().derivative(*args, **kwargs)
