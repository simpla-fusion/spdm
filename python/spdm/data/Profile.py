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


class Profile(Node, Function[_T]):

    def __init__(self, *args, **kwargs) -> None:
        Node.__init__(self, *args, **kwargs)
        if isinstance(self._parent, Container):
            coords = [k for k in self._appinfo.keys() if k.startswith("coordinate")]
            coords.sort()
            coords = [self._appinfo[c] for c in coords]
            coords = [(None if (c == "1...N" or self._entry is None) else self._entry.child(c)) for c in coords]
            Function.__init__(self, *coords, self.__entry__(), appinfo=self._appinfo)
        else:
            raise RuntimeError(f"Parent is None, can not determint the coordinates!")

    @property
    def data(self) -> np.ndarray:
        return super().__value__()
