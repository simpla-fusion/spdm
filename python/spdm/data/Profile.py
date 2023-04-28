import collections.abc
import typing
from enum import Enum
from functools import cached_property
from typing import Any

import numpy as np
from spdm.data.Dict import Dict
from spdm.data.Function import Function
from spdm.data.List import List
from spdm.data.Node import Node
from spdm.data.sp_property import sp_property
from spdm.utils.logger import logger
from spdm.utils.tags import _undefined_

from .Container import Container

_T = typing.TypeVar("_T")


class Profile(Node, Function, typing.Generic[_T]):

    def __init__(self,   *args,   **kwargs) -> None:
        Node.__init__(*args,   **kwargs)
        if isinstance(self._parent, Container):
            coords = [k for k in self._appinfo.keys() if k.startswith("coordinate")]
            coords.sort()
            Function.__init__(*[self._parent[c] for c in coords], self, appinfo=self._appinfo)
        else:
            raise RuntimeError(f"Parent is None, can not determint the coordinates!")

    @property
    def data(self) -> np.ndarray:
        return super().__value__()
