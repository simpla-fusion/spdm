from __future__ import annotations

import inspect
import typing
from copy import copy
from functools import cached_property

import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import (InterpolatedUnivariateSpline, interp1d, interp2d, UnivariateSpline, RectBivariateSpline,
                               RegularGridInterpolator,
                               RectBivariateSpline)
from enum import Enum
from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix
from ..utils.tags import _not_found_
from ..utils.typing import ArrayType, NumericType, numeric_type, scalar_type, array_type
from .Expression import Expression

import collections.abc

_T = typing.TypeVar("_T")


class Function(Expression, typing.Generic[_T]):
    """
        Function
        ---------
        A function is a mapping between two sets, the _mesh_ and the  _value_.
        The _value_  is the set of all possible outputs of the function.
        The _mesh_ is the set of all possible inputs  to the function.

        函数定义域为多维空间时，网格采用rectlinear mesh，即每个维度网格表示为一个数组 _dimension_ 。

    """

    def __init__(self, value: NumericType | Expression, *dims: ArrayType,  mesh=None, name=None, periods=None, **kwargs):
        """
            Parameters
            ----------
            value : NumericType
                函数的值
            mesh : typing.List[ArrayType]
                函数的定义域
            args : typing.Any
                位置参数, 用于与mesh_*，coordinate* 一起构建 mesh
            kwargs : typing.Any
                命名参数，
                    mesh_*      : 用于传递给网格的参数
                    coordinate* : 给出各个坐标轴的path
                    op_*        : 用于传递给运算符的参数
                    *           : 用于传递给 Node 的参数

        """
        if isinstance(value, Expression) or callable(value):
            Expression.__init__(self, value, **kwargs)
            self._value = None
        else:
            Expression.__init__(self,  **kwargs)
            self._value = value

        if mesh is None or mesh is _not_found_:
            self._mesh = dims
        elif isinstance(mesh, collections.abc.Sequence) and all(isinstance(d, array_type) for d in mesh):
            self._mesh = mesh
            if len(dims) > 0:
                logger.warning(f"Function.__init__: mesh is  ignored! {len(dims)} {type(mesh)}")
        elif isinstance(mesh, collections.abc.Mapping):
            self._mesh = mesh
            if len(dims) > 0:
                o_dims = self._mesh.setdefault("dims", dims)
                if o_dims != dims:
                    logger.warning(f"Function.__init__: mesh is dict, dims is ignored! {mesh} {dims}")
        elif isinstance(mesh, Enum):
            self._mesh = {"type": mesh.name, "dims": dims}
        elif isinstance(mesh, str):
            self._mesh = {"type":  mesh, "dims": dims}
        else:
            self._mesh = mesh

        self._name = str(name) if name is not None else f"{self.__class__.__name__}"

        self._periods = periods
        self._ppoly = None

        if self._value is not None:
            self.validate(strict=True)

    @property
    def name(self) -> dict: return self._name

    def validate(self, value=None, strict=False) -> bool:
        """ 检查函数的定义域和值是否匹配 """

        m_shape = tuple(self.shape)

        v_shape = ()

        if value is None:
            value = self.__value__()

        if value is None:
            raise RuntimeError(f" value is None! {self.__str__()}")

        if isinstance(value, array_type):
            v_shape = tuple(value.shape)

        if (v_shape == m_shape) or (v_shape[:-1] == m_shape):
            return True
        elif strict:
            raise RuntimeError(f" value.shape is not match with mesh! {v_shape}!={m_shape} ")
        else:
            logger.warning(f" value.shape is not match with mesh! {v_shape}!={m_shape} ")
            return False

    def __duplicate__(self) -> Function:
        """ 复制一个新的 Function 对象 """
        other: Function = super().__duplicate__()
        other._value = self._value
        other._mesh = self._mesh
        other._name = self._name
        return other

    def __serialize__(self) -> typing.Mapping: raise NotImplementedError(f"")

    @classmethod
    def __deserialize__(cls, desc: typing.Mapping) -> Function: raise NotImplementedError(f"")

    def __str__(self) -> str: return self.name if self._value is not None else super().__str__()

    @property
    def is_empty(self) -> bool: return self._value is None and self._mesh is None and super().is_empty

    @property
    def __type_hint__(self) -> typing.Type:
        """ 获取函数的类型
        """
        orig_class = getattr(self, "__orig_class__", None)

        tp = typing.get_args(orig_class)[0]if orig_class is not None else None

        return tp if inspect.isclass(tp) else float

    @property
    def ndim(self) -> int: return len(self._mesh)
    """ 函数的维度，即定义域的秩 """

    @property
    def rank(self) -> int:
        """ 函数的秩，rank=1 标量函数， rank=3 矢量函数 None 待定 """
        if isinstance(self._value, array_type):
            return self._value.shape[-1]
        elif isinstance(self._value, tuple):
            return len(self._value)
        else:
            logger.warning(f"Function.rank is not defined!  {type(self._value)} default=1")
            return 1

    @property
    def mesh(self) -> typing.List[ArrayType]: return self._mesh
    """ 函数的定义域，即函数的自变量的取值范围。
        每个维度对应一个一维数组，为网格的节点。 """
    @property
    def dimensions(self) -> typing.List[ArrayType]: return self._mesh

    @property
    def dims(self) -> typing.List[ArrayType]: return self._mesh

    @property
    def shape(self) -> typing.Tuple[int]: return tuple(len(d) for d in self._mesh)

    @property
    def periods(self) -> typing.Tuple[float]: return self._periods

    @cached_property
    def bbox(self) -> typing.Tuple[ArrayType, ArrayType]:
        """ bound box 返回包裹函数参数的取值范围的最小多维度超矩形（hyperrectangle） """
        if self.ndim == 1:
            return (np.min(self._mesh), np.max(self._mesh))
        else:
            return (np.asarray([np.min(d) for d in self._mesh], dtype=float),
                    np.asarray([np.max(d) for d in self._mesh], dtype=float))

    @cached_property
    def points(self) -> typing.List[ArrayType]:
        if len(self._mesh) == 1:
            return self._mesh
        else:
            return np.meshgrid(*self._mesh, indexing="ij")

    def __value__(self) -> ArrayType: return self._value
    """ 返回函数的数组 self._value """

    def __array__(self, dtype=None, *args, **kwargs) -> ArrayType:
        """ 重载 numpy 的 __array__ 运算符
                若 self._value 为 array_type 或标量类型 则返回函数执行的结果
        """
        res = self.__value__()

        if res is None or res is _not_found_:
            res = self._value = self.__call__(*self.points)

        if isinstance(res, numeric_type):
            res = np.asarray(res, dtype=self.__type_hint__ if dtype is None else dtype)
        else:
            raise TypeError(f"Function.__array__ is not defined for {type(res)}!")
        return res

    @property
    def __mesh__(self): return self._mesh

    def __getitem__(self, *args) -> NumericType: raise NotImplementedError(f"Function.__getitem__ is not implemented!")

    def __setitem__(self, *args) -> None: raise RuntimeError("Function.__setitem__ is prohibited!")

    def compile(self, *args, force=False, in_place=True,  check_nan=True,   **kwargs) -> Function:
        """ 对函数进行编译，获得较快速的多项式插值

            NOTE：
                - 由 points，value  生成插值函数，并赋值给 self._ppoly。 插值函数相对原始表达式的优势是速度快，缺点是精度低。
                - 当函数为expression时，调用 value = self.__call__(*points) 。
            TODO:
                - 支持 JIT 编译, support JIT compile
                - 优化缓存
                - 支持多维插值
                - 支持多维求导，自动微分 auto diff
            -

        """

        if self._ppoly is not None and not force:
            logger.warning(f"Function.compile() is ignored! {self.__str__()} {self._ppoly}")
            return self

        if self._value is None and self.has_children:
            self._value = self.__call__()

        value = self.__value__() if self._value is None else self._value

        if isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
            value = np.asarray(value)

        if isinstance(value, array_type) and len(value.shape) == 0:
            value = value.item()
        # elif len(value.shape) == 1 and value.shape[0] == 1:
        #     value = value[0]
        # else:
        #     logger.debug(value.shape)

        if isinstance(value, scalar_type) or (isinstance(value, array_type) and value.shape == (1,)):  # 如果value是标量，则直接使用标量
            self._ppoly = value
            return self

        if callable(value):  # 如果value是函数，则直接使用函数
            self._ppoly = value
            return self

        if isinstance(value, np.ndarray) and hasattr(self.__mesh__, "interpolator"):  # 如果value是数组，且mesh有插值函数，则直接使用插值函数
            self._ppoly = self.__mesh__.interpolator(value)
            return self

        #  获得坐标点points，用于构建插值函数

        points = getattr(self, "points", None)

        if points is None:  # for Field
            points = getattr(self.__mesh__, "points")

        if points is not None:
            pass
        elif isinstance(self.__mesh__, tuple) and all(isinstance(d, array_type) for d in self.__mesh__):  # rectlinear mesh for Function
            points = np.meshgrid(*self.__mesh__)
        else:
            raise RuntimeError(f"Can not reslove mesh {type(self.__mesh__)}!")

        if all((d.shape if isinstance(d, array_type) else None) for d in points):  # 检查 points 的维度是否一致
            m_shape = points[0].shape
        else:
            raise RuntimeError(f"Function.compile() incorrect points  shape  {self.__str__()} {points}")

        if value is None or value is _not_found_:  # 如果value是None或者_not_found_，且self._ppoly不为空，则调用函数__call__
            raise RuntimeError(f"Function.compile() incorrect value {self.__str__()} value={value} op={self._ppoly} ")

        if not isinstance(value, array_type):
            value = np.asarray(value)
            # raise RuntimeError(f"Function.compile() incorrect value type {self.__str__()} {value}")

        if tuple(value.shape) != tuple(m_shape):
            raise RuntimeError(
                f"Function.compile() incorrect value shape {value.shape}!={m_shape}! value={value} func={self.__str__()} ")

        if len(m_shape) == 1:
            x = points[0]
            if check_nan:
                mark = np.isnan(value)
                nan_count = np.count_nonzero(mark)
                if nan_count > 0:
                    logger.warning(
                        f"{self.__class__.__name__}[{self.__str__()}]: Ignore {nan_count} NaN at {np.argwhere(mark)}.")
                    value = value[~mark]
                    x = x[~mark]

            self._ppoly = InterpolatedUnivariateSpline(x, value)
        elif self.ndim == 2:
            if check_nan:
                mark = np.isnan(value)
                nan_count = np.count_nonzero(mark)
                if nan_count > 0:
                    logger.warning(
                        f"{self.__class__.__name__}[{self.__str__()}]: Replace  {nan_count} NaN by 0 at {np.argwhere(mark)}.")
                    value[mark] = 0.0

            x, y = self._mesh

            if isinstance(self._periods, collections.abc.Sequence):
                logger.warning(f"TODO: periods={self._periods}")

            self._ppoly = RectBivariateSpline(x, y, value), {"grid": False}

        else:
            raise NotImplementedError(
                f"Multidimensional interpolation for n>2 is not supported.! ndim={self.ndim} ")

        return self

    def _fetch_op(self):
        """ 

        """
        if self._ppoly is not None:
            return self._ppoly
        elif not self.callable:
            self.compile()

        if self._ppoly is None:
            return super()._fetch_op()
        else:
            return self._ppoly

    def __call__(self, *args, **kwargs) -> _T | ArrayType:
        """  重载函数调用运算符
            Parameters
            ----------
            args : typing.Any
                位置参数,
            kwargs : typing.Any
        """
        if len(args) == 0:
            args = self.points
        return super().__call__(*args,  **kwargs)

    def derivative(self, n=1) -> Function:
        if self._ppoly is None:
            self.compile()

        if isinstance(self._ppoly, tuple):
            ppoly, opts = self._ppoly
        else:
            ppoly = self._ppoly
            opts = {}

        if self.ndim == 1:
            return Function[_T](None, op=(ppoly.derivative(n), opts),  mesh=self._mesh,  # periods=self.periods,
                                name=f"d_{n}({self.__str__()})")
        elif self.ndim == 2 and n == 1:
            return Function[typing.Tuple[_T, _T]]((ppoly.partial_derivative(1, 0),
                                                  ppoly.partial_derivative(0, 1)),
                                                  mesh=self.mesh,  # periods=self.periods,
                                                  name=f"d{list(n)}({self.__str__()})", **opts)
        elif self.ndim == 3 and n == 1:
            return Function[typing.Tuple[_T, _T, _T]]((ppoly.partial_derivative(1, 0, 0),
                                                       ppoly.partial_derivative(0, 1, 0),
                                                       ppoly.partial_derivative(0, 0, 1)),
                                                      mesh=self.mesh,  # periods=self.periods,
                                                      name=f"d{list(n)}({self.__str__()})", **opts)
        elif self.ndim == 2 and n == 2:
            return Function[typing.Tuple[_T, _T, _T]]((ppoly.partial_derivative(2, 0),
                                                       ppoly.partial_derivative(0, 2),
                                                       ppoly.partial_derivative(1, 1)),
                                                      mesh=self.mesh,  # periods=self.periods,
                                                      name=f"d{list(n)}({self.__str__()})", **opts)
        else:
            raise NotImplemented(f"TODO: ndim={self.ndim} n={n}")

    def d(self, *n) -> Function[_T]: return self.derivative(*n)

    def partial_derivative(self, *n) -> Function:
        if self._ppoly is None:
            self.compile()

        if isinstance(self._ppoly, tuple):
            ppoly, opts = self._ppoly
            op = (ppoly.partial_derivative(*n), opts)
        else:
            op = ppoly.partial_derivative(*n)

        return Function[_T](None, op=op,
                            mesh=self._mesh, periods=self.periods,
                            name=f"d{list(n)}({self.__str__()})")

    def pd(self, *n) -> Function[_T]: return self.partial_derivative(*n)

    def antiderivative(self, *n) -> Function[_T]:
        if self._ppoly is None:
            self.compile()

        if isinstance(self._ppoly, tuple):
            ppoly, opts = self._ppoly
            op = (ppoly.antiderivative(*n), opts)
        else:
            op = self._ppoly.antiderivative(*n)

        return Function[_T](None, op=op,  mesh=self._mesh,  # periods=self.periods,
                            name=f"antiderivative{list(n) if len(n)>0 else ''}({self.__str__()})")

    def dln(self) -> Function[_T]: return self.derivative() / self

    def integral(self, *args, **kwargs) -> _T:
        if self._ppoly is None:
            self.compile()
        return self._ppoly.integral(*args, **kwargs)

    def roots(self, *args, **kwargs) -> _T:
        if self._ppoly is None:
            self.compile()
        return self._ppoly.roots(*args, **kwargs)


def function_like(y: NumericType, *args: NumericType, **kwargs) -> Function:
    if len(args) == 0 and isinstance(y, Function):
        return y
    else:
        return Function(y, *args, **kwargs)


class Piecewise(Expression, typing.Generic[_T]):
    """ PiecewiseFunction
        ----------------
        A piecewise function. 一维或多维，分段函数
    """

    def __init__(self, func: typing.List[typing.Callable], cond: typing.List[typing.Callable], **kwargs):
        super().__init__(op=(func, cond), **kwargs)

    @property
    def rank(self): return 1

    @property
    def ndim(self): return 1

    def _compile(self): return self, {}

    def _apply(self, func, x, *args, **kwargs):
        if isinstance(x, array_type) and isinstance(func, numeric_type):
            value = np.full(x.shape, func)
        elif isinstance(x, numeric_type) and isinstance(func, numeric_type):
            value = func
        elif callable(func):
            value = func(x)
        else:
            raise ValueError(f"PiecewiseFunction._apply() error! {func} {x}")
            # [(node(*args, **kwargs) if callable(node) else (node.__entry__().__value__() if hasattr(node, "__entry__") else node))
            #          for node in self._expr_nodes]
        return value

    def __call__(self, x, *args, **kwargs) -> NumericType:
        if isinstance(x, float):
            res = [self._apply(fun, x) for fun, cond in zip(*self._op) if cond(x)]
            if len(res) == 0:
                raise RuntimeError(f"Can not fit any condition! {x}")
            elif len(res) > 1:
                raise RuntimeError(f"Fit multiply condition! {x}")
            return res[0]
        elif isinstance(x, array_type):
            res = np.hstack([self._apply(fun, x[cond(x)]) for fun, cond in zip(*self._op)])
            if len(res) != len(x):
                raise RuntimeError(f"PiecewiseFunction result length not equal to input length, {len(res)}!={len(x)}")
            return res
        else:
            raise TypeError(f"PiecewiseFunction only support single float or  1D array, {x}")
