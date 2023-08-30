from __future__ import annotations

import collections.abc
import typing

from spdm.data.Entry import Entry
from spdm.data.HTree import HTree


from ..utils.tags import _not_found_
from .HTree import Dict


class NamedDict(Dict):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __getattr__(self, key) -> typing.Any:
        value = super()._get(key, _not_found_)

        if value is _not_found_:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute {key}")

        if isinstance(value, collections.abc.Mapping):
            return NamedDict(value, parent=self)
        else:
            return value

    def __setattr__(self, key, value) -> None:
        """ 以 '_' 开始的 key 视为属性，其他的 key 视为字典键值。"""
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            super().update(key, value)

    def __delattr__(self, key: str) -> None:
        """ 以 '_' 开始的 key 视为属性，其他的 key 视为字典键值。"""
        if key.startswith("_"):
            super().__delattr__(key)
        else:
            super().remove(key)
