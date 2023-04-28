from typing import Any
import numpy as np
import collections.abc
from enum import Enum

from spdm.data.Node import Node
from spdm.data.Function import Function
from spdm.data.List import List
from spdm.data.Dict import Dict
from spdm.data.sp_property import sp_property

from spdm.utils.logger import logger
import typing

_T = typing.TypeVar("_T")


class Profile(Node, typing.Generic[_T]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = None

    def __value__(self) -> Function:
        if self._cahce is None:
            y = super().__value__()
            self._cahce = Function()
        return self._cache

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
