from __future__ import annotations

import typing

from spdm.data.HTree import HTreeNode

from .List import List
from .Node import Node
from ..utils.tags import _not_found_

_T = typing.TypeVar("_T")


class HTreeNode(Node, typing.Generic[_T]):

    def __init__(self, *args, **kwargs) -> None:
        pass


class HTree(typing.List[_T]):
    """ hierarchical Tree"""

    def _as_child(self,
                  key: typing.Union[int, str, slice, None],
                  value=_not_found_,
                  type_hint: typing.Type = None,
                  default_value: typing.Any = _not_found_,
                  getter=None,
                  strict=True,
                  **kwargs) -> _T | HTree[_T]:

        return res
