from __future__ import annotations

import collections.abc
import typing


from ..utils.tags import _not_found_
from .HTree import Dict


class NamedDict(Dict):
    def __getitem__(self, key) -> typing.Any:
        return super().__getitem__(key)

    def __getattr__(self, key) -> typing.Any:
        if isinstance(self._cache, dict):
            value = self._cache.get(key, _not_found_)
        else:
            value = _not_found_

        if value is _not_found_:
            value = self._entry.get(key, _not_found_)

        if value is _not_found_:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute {key}")

        if isinstance(value, collections.abc.Mapping):
            return NamedDict(value, parent=self)
        else:
            return value

    def __setattr__(self, key, value) -> None:
        self._cache[key] = value

    def __delattr__(self, key: str) -> None:
        if isinstance(self._cache, dict) and key in self._cache:
            del self._cache[key]
        self._entry.child(key).remove()
