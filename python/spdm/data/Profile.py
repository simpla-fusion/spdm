import typing
import pprint

from ..utils.misc import group_dict_by_prefix
from ..utils.typing import ArrayType
from .Container import Container
from .Function import Function, Expression
from .Node import Node

_T = typing.TypeVar("_T")


class Profile(Node, Expression[_T]):

    def __init__(self,  *args, **kwargs) -> None:
        coordinates, kwargs = group_dict_by_prefix(kwargs,   "coordinate")

        if len(args) > 0 and isinstance(args[0], Expression):            
            expr = args[0]
            args = args[1:]
            super().__init__(*args, **kwargs)
            self.__duplicate_from__(expr)
        else:
            super().__init__(*args, **kwargs)
            Function.__init__(self, None)

        if not isinstance(self._parent, Container):
            raise RuntimeError(f"Parent is None, can not determint the coordinates!")

        if len(coordinates) > 0:
            coord_keys = [*coordinates.keys()]
            coord_keys.sort()
            coord_keys = [coordinates[c] for c in coord_keys]

            # FIXME: "1...N" is for IMAS dd
            self._mesh = [(slice(None) if (c == "1...N") else self._find_node_by_path(c)) for c in coord_keys]

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}>{pprint.pformat(self.__value__())}</{self.__class__.__name__}>"

    def __value__(self) -> ArrayType: return Function.__array__(self)
