from __future__ import annotations

import typing
from enum import Enum

import xmlschema

from ..utils.numeric import array_type
from ..utils.plugin import Pluggable
from ..utils.tags import _not_found_
from .List import List
from .Node import Node

_T = typing.TypeVar("_T")


class Schema(Pluggable):

    def __init__(self, schema_path, *args, **kwargs) -> None:
        self._schema = xmlschema.XMLSchema(schema_path)
        pass

    def validate(self, d, *args, **kwargs) -> bool:
        """
            Validate 和 Verify 是两个相关但不同的概念。
            Verify，中文翻译叫“验证”，就是把事情做对（do things right）;
            Validate，中文翻译叫“确认”，就是做对的事情（do right things)。
            简单来说，Verify 是从内部测试的角度，而 Validate 是从外部用户的角度。
        """
        
        return self._schema.is_valid(d)
