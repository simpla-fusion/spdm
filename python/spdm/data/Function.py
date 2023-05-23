from __future__ import annotations

import inspect
import typing
from copy import copy
from functools import cached_property

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from enum import Enum
from ..utils.logger import logger
from ..utils.misc import group_dict_by_prefix
from ..utils.tags import _not_found_
from ..utils.typing import ArrayType, NumericType, numeric_types
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

    def __init__(self, value: NumericType | Expression, *dims: ArrayType,  mesh=None, name=None, cycles=None, **kwargs):
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
        elif isinstance(mesh, collections.abc.Sequence) and all(isinstance(d, np.ndarray) for d in mesh):
            self._mesh = mesh
            if len(dims) > 0:
                raise RuntimeError(f"Function.__init__: mesh is  ignored! {dims} {mesh}")
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

        self._cycles = cycles

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

        if isinstance(value, np.ndarray):
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
        if isinstance(self.__value__(), np.ndarray):
            return self.__value__().shape[-1]
        elif isinstance(self.__value__(), tuple):
            return len(self.__value__())
        else:
            logger.warning(f"Function.rank is not defined!  {type(self.__value__())} default=1")
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
    def cycles(self) -> typing.Tuple[float]: return self.metadata.get("cycles", [])

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

    def __array__(self) -> ArrayType:
        """ 重载 numpy 的 __array__ 运算符
                若 self._value 为 np.ndarray 或标量类型 则返回函数执行的结果
        """
        res = self.__value__()
        if not isinstance(res, numeric_types):
            res = self.__call__()

        if not isinstance(res, np.ndarray):
            res = np.asarray(res, dtype=self.__type_hint__)

        return res

    def __getitem__(self, *args) -> NumericType: return self.__array__().__getitem__(*args)

    def __setitem__(self, *args) -> None: raise RuntimeError("Function.__setitem__ is prohibited!")

    def _compile(self, in_place=True, check_nan=True):
        """ create op if not exists
            if in_place then update self._op  else do not change current object
           TODO:
            - support JIT compile
            - 优化缓存
            - 支持多维插值
            - 支持多维求导，自动微分 auto diff
            -
        """
        if self.is_epxression:
            value = self.__call__(*self.points)
        elif self._op is None:
            value = self.__value__()
        else:
            return self._op

        if value is None:
            raise RuntimeError(
                f"Function._eval(): self._op is None and self._value is None! {self.__str__()} {self._op} {self._value}")
        elif hasattr(self.mesh, "interpolator"):
            op = self.mesh.interpolator(value)
        elif isinstance(self._mesh, tuple) and len(self._mesh) > 0:
            if isinstance(self._mesh[0], (int, float)) or (isinstance(self._mesh[0], np.ndarray) and len(self._mesh[0]) == 1):
                ppoly = lambda *_, _v=value: _v
                opts = {}
            elif self.ndim == 1:
                x, *_ = self.points
                if check_nan:
                    mark = np.isnan(value)
                    nan_count = np.count_nonzero(mark)
                    if nan_count > 0:
                        logger.warning(
                            f"{self.__class__.__name__}[{self.__str__()}]: Ignore {nan_count} NaN at {np.argwhere(mark)}.")
                        value = value[~mark]
                        x = x[~mark]
                ppoly = InterpolatedUnivariateSpline(x, value)
                opts = {}
            elif self.ndim == 2 and all(isinstance((d, np.ndarray) and d.ndim == 1) for d in self._mesh):
                if check_nan:
                    mark = np.isnan(value)
                    nan_count = np.count_nonzero(mark)
                    if nan_count > 0:
                        logger.warning(f"{self.__class__.__name__}[{self.__str__()}]: Replace  {nan_count} NaN by 0 at {np.argwhere(mark)}.")
                        value[mark] = 0.0

                ppoly = RectBivariateSpline(*self._mesh, value)
                opts = {"grid": False}
            else:
                raise NotImplementedError(
                    f"Multidimensional interpolation for n>2 is not supported.! ndim={self.ndim} ")

            op = ppoly, opts
        else:
            raise RuntimeError(f"Can not create op from {self._mesh}")

        if in_place or self._op is None:
            self._op = op
            self._expr_nodes = ()

        return op

    def compile(self) -> Function:
        """ 编译函数，返回一个新的(加速的)函数对象 """
        return self.__class__(op=self._compile(in_place=False), mesh=self.mesh, cycles=self._cycles, name=f"[{self.__str__()}]")

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

        if self.is_epxression:
            return super().__call__(*args,  **kwargs)

        _, opts = self._compile()

        if all([isinstance(a, np.ndarray) for a in args]) and not opts.get("grid", True):
            shape = args[0].shape
            return super().__call__(*[a.ravel() for a in args],  **kwargs).reshape(shape)
        else:
            return super().__call__(*args,  **kwargs)

    def derivative(self, n=1) -> Function:
        ppoly, _, opts = self._compile()

        if self.ndim == 1 and n == 1:
            return Function[_T](op=ppoly.derivative(*n),  mesh=self._mesh, cycles=self.cycles,
                                name=f"d{list(n)}({self.__str__()})", **opts)
        elif self.ndim == 2 and n == 1:
            return Function[typing.Tuple[_T, _T]]((ppoly.partial_derivative(1, 0),
                                                  ppoly.partial_derivative(0, 1)),
                                                  mesh=self.mesh, cycle=self.cycles,
                                                  name=f"d{list(n)}({self.__str__()})", **opts)
        elif self.ndim == 3 and n == 1:
            return Function[typing.Tuple[_T, _T, _T]]((ppoly.partial_derivative(1, 0, 0),
                                                       ppoly.partial_derivative(0, 1, 0),
                                                       ppoly.partial_derivative(0, 0, 1)),
                                                      mesh=self.mesh, cycle=self.cycles,
                                                      name=f"d{list(n)}({self.__str__()})", **opts)
        elif self.ndim == 2 and n == 2:
            return Function[typing.Tuple[_T, _T, _T]]((ppoly.partial_derivative(2, 0),
                                                       ppoly.partial_derivative(0, 2),
                                                       ppoly.partial_derivative(1, 1)),
                                                      mesh=self.mesh, cycle=self.cycles,
                                                      name=f"d{list(n)}({self.__str__()})", **opts)
        else:
            raise NotImplemented(f"TODO: ndim={self.ndim} n={n}")

    def d(self) -> Function[_T]: return self.derivative()

    def partial_derivative(self, *n) -> Function:
        ppoly,  opts = self._compile()
        return Function[_T](ppoly.partial_derivative(*n), mesh=self._mesh, cycles=self.cycles,
                            name=f"d{list(n)}({self.__str__()})", **opts)

    def pd(self, *n) -> Function[_T]: return self.partial_derivative(*n)

    def antiderivative(self, *n) -> Function[_T]:
        ppoly, opts = self._compile()
        return Function[_T](ppoly.antiderivative(*n),  mesh=self._mesh,  # cycles=self.cycles,
                            name=f"antiderivative{list(n) if len(n)>0 else ''}({self.__str__()})", **opts)

    def dln(self) -> Function[_T]: return self.derivative() / self

    def integral(self, *args, **kwargs) -> _T:
        ppoly, opts = self._compile()
        return ppoly.integral(*args, **kwargs)

    def roots(self, *args, **kwargs) -> _T:
        ppoly, opts = self._compile()
        return ppoly.roots(*args, **kwargs)


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

    def _apply(self, func, x, *args, **kwargs):
        if isinstance(x, np.ndarray) and isinstance(func, numeric_types):
            value = np.full(x.shape, func)
        elif isinstance(x, numeric_types) and isinstance(func, numeric_types):
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
        elif isinstance(x, np.ndarray):
            res = np.hstack([self._apply(fun, x[cond(x)]) for fun, cond in zip(*self._op)])
            if len(res) != len(x):
                raise RuntimeError(f"PiecewiseFunction result length not equal to input length, {len(res)}!={len(x)}")
            return res
        else:
            raise TypeError(f"PiecewiseFunction only support single float or  1D array, {x}")
