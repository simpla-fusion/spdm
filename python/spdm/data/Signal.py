import typing


from ..utils.typing import ArrayType
from .Field import Field
from .Profile import Profile

_T = typing.TypeVar("_T")


class Signal(Profile[_T]):
    """Signal with its time base
    """

    @property
    def data(self) -> ArrayType: return super().__array__()

    @property
    def time(self) -> ArrayType: return super().domain[0]


class SignalND(Field[_T]):
    """Signal with its time base
    """

    pass
