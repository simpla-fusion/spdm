import typing

from ..utils.numeric import bitwise_and, float_nan
from ..utils.typing import NumericType, array_type, numeric_type
from .Expression import Expression


class Domain(Expression):
    """
        定义空间中的一个区域

        ```{python}
        >>>d= Domain(0<x,x<1)
        >>>d(0.5)
        True
        >>>d(1.2)
        False
        ```
    """

    def __init__(self, *cond, fill_value: NumericType = None, **kwargs) -> None:

        if len(cond) == 0:
            Expression.__init__(self, True)
        elif len(cond) == 1:
            Expression.__init__(self, cond[0])
        else:
            Expression.__init__(self, lambda *d: bitwise_and.reduce(d), *cond)

        self._fill_value = fill_value if fill_value is not None else float_nan

    def mask_like(self, target: array_type, *args: NumericType, fill_value=None) -> array_type:
        if not isinstance(target, array_type):
            raise TypeError(f"target should be array_type not {type(target)}")

        mask = self.__call__(*args)

        if isinstance(mask, array_type):
            target[mask] = fill_value if fill_value is not None else self._fill_value

        return target
